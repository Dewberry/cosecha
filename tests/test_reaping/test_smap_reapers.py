"""Tests for SMAP L4 reaper implementation."""

from __future__ import annotations

import io
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cosecha.exceptions import (
    APIError,
    DataNotFoundError,
    DateRangeError,
    ReaperError,
    TransformationError,
)
from cosecha.reaping.smap import SMAPL4Reaper

pytest.importorskip("earthaccess")

# Mock grid: latitude descends with row, longitude ascends with column (as on EASE-Grid).
LAT = np.array([35.0, 34.0, 33.0, 32.0, 31.0])
LON = np.array([-100.0, -99.0, -98.0, -97.0, -96.0, -95.0])
BBOX = {"spatial_subset": {"lat_bounds": (31.5, 34.0), "lon_bounds": (-98.5, -96.0)}}
DATES = ("2026-09-28", "2026-09-28T06:00")


def _granule(begin: str) -> dict:
    """Build a minimal earthaccess DataGranule-like dict for a 3-hour granule."""
    start = pd.Timestamp(begin)
    return {
        "umm": {
            "TemporalExtent": {
                "RangeDateTime": {
                    "BeginningDateTime": f"{start:%Y-%m-%dT%H:%M:%SZ}",
                    "EndingDateTime": f"{start + pd.Timedelta(hours=3):%Y-%m-%dT%H:%M:%SZ}",
                }
            },
            "RelatedUrls": [
                {"URL": f"https://data/{begin}.h5", "Type": "GET DATA"},
                {"URL": f"https://opendap/{start:%H}.h5", "Subtype": "OPENDAP DATA"},
            ],
        }
    }


def _response(ds: xr.Dataset, group: str | None = None) -> SimpleNamespace:
    buf = io.BytesIO()
    ds.to_netcdf(buf, engine="h5netcdf", group=group)
    return SimpleNamespace(content=buf.getvalue(), raise_for_status=lambda: None)


def _opendap_get(url, params, timeout):
    """Mimic the OPeNDAP DAP4 .dap.nc4 endpoint for the mock grid."""
    ce = params["dap4.ce"]
    if "cell_lat" in ce:
        return _response(xr.Dataset({"cell_lat": (("r", "c"), LAT[:, None])}))
    if "cell_lon" in ce:
        return _response(xr.Dataset({"cell_lon": (("r", "c"), LON[None])}))
    r0, r1, c0, c1 = map(int, re.search(r"\[(\d+):1:(\d+)\]\[(\d+):1:(\d+)\]", ce).groups())
    data = np.full(
        (r1 - r0 + 1, c1 - c0 + 1), float(re.search(r"/(\d\d)\.h5", url).group(1)), dtype="float32"
    )
    data[0, 0] = np.nan
    variables = re.findall(r"/Geophysical_Data/(\w+)\[", ce)
    ds = xr.Dataset({v: (("y", "x"), data, {"units": "m3 m-3"}) for v in variables})
    for v in variables:
        ds[v].encoding["_FillValue"] = -9999.0
    return _response(ds, group="Geophysical_Data")


@pytest.fixture
def session(mocker):
    sess = MagicMock()
    sess.get.side_effect = _opendap_get
    mocker.patch.object(SMAPL4Reaper, "_login", return_value=sess)
    return sess


@pytest.fixture
def search(mocker):
    granules = [_granule(t) for t in ("2026-09-27T21:00", "2026-09-28T00:00", "2026-09-28T03:00")]
    return mocker.patch("earthaccess.search_data", return_value=granules)


class TestSMAPL4Reaper:
    """Test SMAPL4Reaper implementation."""

    def test_initialization(self):
        reaper = SMAPL4Reaper(dates=DATES, variables=["sm_surface"], transformations=BBOX)
        assert reaper.is_latest is False
        assert reaper.start_date == pd.Timestamp("2026-09-28", tz="UTC")
        assert reaper.bounds == BBOX["spatial_subset"]
        assert reaper.transformations == {}

    def test_initialization_latest(self):
        reaper = SMAPL4Reaper(dates="latest")
        assert reaper.is_latest is True
        assert reaper.start_date < reaper.end_date

    def test_invalid_dates(self):
        with pytest.raises(ValueError, match="dates must be"):
            SMAPL4Reaper(dates="2026-09-28")  # type: ignore
        with pytest.raises(DateRangeError):
            SMAPL4Reaper(dates=("2026-09-29", "2026-09-28"))

    def test_empty_variables(self):
        with pytest.raises(ReaperError, match="at least one variable"):
            SMAPL4Reaper(dates="latest", variables=[])

    def test_unknown_variable(self):
        with pytest.raises(ReaperError, match=r"Unknown SPL4SMGP variables \['sm_surfce'\]"):
            SMAPL4Reaper(dates="latest", variables=["sm_surface", "sm_surfce"])

    @pytest.mark.parametrize(("max_workers", "timeout"), [(0, 120), (4, 0)])
    def test_invalid_workers_or_timeout(self, max_workers, timeout):
        with pytest.raises(ReaperError, match="must be positive"):
            SMAPL4Reaper(dates="latest", max_workers=max_workers, timeout=timeout)

    def test_unknown_spatial_subset_key(self):
        with pytest.raises(
            TransformationError, match=r"Unknown spatial_subset keys \['lat_bound'\]"
        ):
            SMAPL4Reaper(
                dates="latest", transformations={"spatial_subset": {"lat_bound": (30, 40)}}
            )

    @pytest.mark.parametrize(
        "subset",
        [
            {"lat_bounds": (34, 31)},
            {"lat_bounds": (-95, 0)},
            {"lon_bounds": (170, -170)},
            {"lon_bounds": (0, 200)},
            {"lon_bounds": (-100,)},
        ],
    )
    def test_invalid_bounds(self, subset):
        with pytest.raises(TransformationError, match="must be \\(min, max\\)"):
            SMAPL4Reaper(dates="latest", transformations={"spatial_subset": subset})

    def test_find_granules_filters_by_center_time(self, search):
        result = SMAPL4Reaper(dates=DATES)._find_granules()

        assert [t for t, _ in result] == [
            pd.Timestamp("2026-09-28 01:30", tz="UTC"),
            pd.Timestamp("2026-09-28 04:30", tz="UTC"),
        ]
        assert result[0][1] == "https://opendap/00.h5"
        assert search.call_args.kwargs["short_name"] == "SPL4SMGP"

    def test_find_granules_latest(self, search):
        reaper = SMAPL4Reaper(dates="latest")
        reaper.start_date = pd.Timestamp("2026-09-27", tz="UTC")

        assert [t for t, _ in reaper._find_granules()] == [
            pd.Timestamp("2026-09-28 04:30", tz="UTC")
        ]

    def test_find_granules_none(self, mocker):
        mocker.patch("earthaccess.search_data", return_value=[])
        with pytest.raises(DataNotFoundError, match="No SPL4SMGP granules"):
            SMAPL4Reaper(dates=DATES)._find_granules()

    def test_find_granules_search_error(self, mocker):
        mocker.patch("earthaccess.search_data", side_effect=RuntimeError("CMR down"))
        with pytest.raises(APIError, match="Could not search NASA CMR"):
            SMAPL4Reaper(dates=DATES)._find_granules()

    def test_reap_with_spatial_subset(self, session, search):
        reaper = SMAPL4Reaper(
            dates=DATES,
            variables=["sm_surface", "sm_rootzone"],
            transformations={**BBOX, "variable_rename": {"sm_surface": "sm"}},
        )

        ds = reaper.reap()

        assert set(ds.data_vars) == {"sm", "sm_rootzone"}
        assert dict(ds.sizes) == {"time": 2, "latitude": 3, "longitude": 3}
        np.testing.assert_array_equal(ds.latitude, [32.0, 33.0, 34.0])
        np.testing.assert_array_equal(ds.longitude, [-98.0, -97.0, -96.0])
        assert ds.time.values[0] == np.datetime64("2026-09-28T01:30")
        assert ds["sm"].attrs["units"] == "m3 m-3"
        # Fill value at the north-west corner becomes NaN after flipping to ascending lat.
        assert np.isnan(ds["sm"].sel(latitude=34.0, longitude=-98.0)).all()
        assert ds["sm"].isel(time=1, latitude=0, longitude=0).item() == 3.0
        # Two coordinate requests plus one server-side subset request per granule.
        assert session.get.call_count == 4
        assert session.get.call_args.kwargs["params"]["dap4.ce"] == (
            "/Geophysical_Data/sm_surface[1:1:3][2:1:4];/Geophysical_Data/sm_rootzone[1:1:3][2:1:4]"
        )

    def test_reap_global(self, session, search):
        ds = SMAPL4Reaper(dates=DATES, variables=["sm_surface"]).reap()

        assert dict(ds.sizes) == {"time": 2, "latitude": 5, "longitude": 6}
        assert bool((ds.latitude.diff("latitude") > 0).all())

    def test_subset_without_cells(self, session, search):
        reaper = SMAPL4Reaper(
            dates=DATES, transformations={"spatial_subset": {"lat_bounds": (33.2, 33.8)}}
        )
        with pytest.raises(DataNotFoundError, match="contains no SMAP cells"):
            reaper.reap()

    def test_reap_wraps_request_errors(self, session, search):
        session.get.side_effect = ConnectionError("reset")
        with pytest.raises(ReaperError, match="SMAP L4 reaping failed: reset"):
            SMAPL4Reaper(dates=DATES).reap()

    @pytest.fixture
    def no_credentials(self, monkeypatch, tmp_path):
        for var in ("EARTHDATA_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("NETRC", str(tmp_path / "netrc"))
        return tmp_path / "netrc"

    def test_login_without_credentials(self, no_credentials):
        with pytest.raises(APIError, match="credentials not found"):
            SMAPL4Reaper(dates="latest")._login()

    def test_login_token(self, no_credentials, monkeypatch):
        monkeypatch.setenv("EARTHDATA_TOKEN", "abc")
        session = SMAPL4Reaper(dates="latest")._login()
        assert session.headers["Authorization"] == "Bearer abc"

    def test_login_username_password(self, no_credentials, monkeypatch, mocker):
        monkeypatch.setenv("EARTHDATA_USERNAME", "user")
        monkeypatch.setenv("EARTHDATA_PASSWORD", "p!w:)")
        session_cls = mocker.patch("earthaccess.auth.SessionWithHeaderRedirection")
        SMAPL4Reaper(dates="latest")._login()
        session_cls.assert_called_once_with("urs.earthdata.nasa.gov", ("user", "p!w:)"))

    def test_login_netrc(self, no_credentials, mocker):
        no_credentials.write_text("machine urs.earthdata.nasa.gov login user password pw\n")
        no_credentials.chmod(0o600)
        session_cls = mocker.patch("earthaccess.auth.SessionWithHeaderRedirection")
        SMAPL4Reaper(dates="latest")._login()
        session_cls.assert_called_once_with("urs.earthdata.nasa.gov", ("user", "pw"))

    @pytest.mark.network
    def test_reap_network(self):
        """Live fetch; requires NASA Earthdata credentials."""
        start = pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=7)
        reaper = SMAPL4Reaper(
            dates=(str(start), str(start + pd.Timedelta(hours=6))),
            variables=["sm_surface", "sm_rootzone"],
            transformations={
                "spatial_subset": {"lat_bounds": (31.5, 34.0), "lon_bounds": (-98.5, -95.5)}
            },
        )
        ds = reaper.reap()
        assert ds.sizes["time"] == 2
        assert float(ds.latitude.min()) >= 31.5
        assert float(ds.longitude.max()) <= -95.5
        assert float(ds["sm_surface"].max()) <= 1.0
