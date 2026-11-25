"""Tests for SMAP L4 reaper implementation."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from aiohttp import ClientRequest, ClientResponseError
from tiny_retriever.exceptions import ServiceError
from yarl import URL

from cosecha.exceptions import (
    APIError,
    DataNotFoundError,
    DateRangeError,
    ReaperError,
    TransformationError,
)
from cosecha.reaping.smap import _CMR_URL, SMAPL4Reaper, _EarthdataAuth, _netrc_credentials

CREDENTIAL_VARS = ("EARTHDATA_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")

# Mock grid: latitude descends with row, longitude ascends with column (as on EASE-Grid).
LAT = np.array([35.0, 34.0, 33.0, 32.0, 31.0])
LON = np.array([-100.0, -99.0, -98.0, -97.0, -96.0, -95.0])
BBOX = {"spatial_subset": {"lat_bounds": (31.5, 34.0), "lon_bounds": (-98.5, -96.0)}}
DATES = ("2026-09-28", "2026-09-28T06:00")
OPENDAP = "https://opendap.earthdata.nasa.gov/collections/C1/granules"


def _granule(begin: str, opendap: str = OPENDAP) -> dict:
    """Build a minimal CMR UMM-JSON item for a 3-hour granule."""
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
                {"URL": f"{opendap}/{start:%H}.h5", "Subtype": "OPENDAP DATA"},
            ],
        }
    }


GRANULES = [_granule(t) for t in ("2026-09-27T21:00", "2026-09-28T00:00", "2026-09-28T03:00")]


def _netcdf(ds: xr.Dataset, group: str | None = None) -> bytes:
    buf = io.BytesIO()
    ds.to_netcdf(buf, engine="h5netcdf", group=group)
    return buf.getvalue()


def _opendap(url: str, ce: str) -> bytes:
    """Mimic the OPeNDAP DAP4 .dap.nc4 endpoint for the mock grid."""
    if "cell_lat" in ce:
        return _netcdf(xr.Dataset({"cell_lat": (("r", "c"), LAT[:, None])}))
    if "cell_lon" in ce:
        return _netcdf(xr.Dataset({"cell_lon": (("r", "c"), LON[None])}))
    r0, r1, c0, c1 = map(int, re.search(r"\[(\d+):1:(\d+)\]\[(\d+):1:(\d+)\]", ce).groups())
    data = np.full(
        (r1 - r0 + 1, c1 - c0 + 1),
        float(re.search(r"/(\d\d)\.h5\.dap\.nc4$", url).group(1)),
        dtype="float32",
    )
    data[0, 0] = np.nan
    variables = re.findall(r"/Geophysical_Data/(\w+)\[", ce)
    ds = xr.Dataset({v: (("y", "x"), data, {"units": "m3 m-3"}) for v in variables})
    for v in variables:
        ds[v].encoding["_FillValue"] = -9999.0
    return _netcdf(ds, group="Geophysical_Data")


def _fake_fetch(urls, return_type, *, request_kwargs, **_):
    """Stand in for tiny_retriever.fetch, serving CMR pages and OPeNDAP subsets."""
    if urls[0] == _CMR_URL:
        return [{"hits": len(GRANULES), "items": GRANULES} for _ in urls]
    return [
        _opendap(u, kw["params"]["dap4.ce"]) for u, kw in zip(urls, request_kwargs, strict=True)
    ]


def _service_error(status: int) -> ServiceError:
    err = ServiceError(f"{status}", OPENDAP)
    err.__cause__ = ClientResponseError(MagicMock(), (), status=status)
    return err


@pytest.fixture
def isolated_netrc(monkeypatch, tmp_path):
    path = tmp_path / ".netrc"
    monkeypatch.setenv("NETRC", str(path))
    return path


@pytest.fixture
def netrc_file(isolated_netrc):
    isolated_netrc.write_text("machine urs.earthdata.nasa.gov login netrc-user password netrc-pw\n")
    isolated_netrc.chmod(0o600)
    return isolated_netrc


@pytest.fixture
def credentials(monkeypatch, isolated_netrc):
    for var in CREDENTIAL_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("EARTHDATA_USERNAME", "user")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "p!w:)")


@pytest.fixture
def fetch(mocker, credentials):
    return mocker.patch("cosecha.reaping.smap.tiny_retriever.fetch", side_effect=_fake_fetch)


class TestSMAPL4ReaperNetwork:
    """Live tests against NASA CMR and OPeNDAP."""

    @pytest.mark.network
    def test_reap_network(self):
        """Live fetch; requires NASA Earthdata credentials."""
        if not (
            os.getenv("EARTHDATA_TOKEN")
            or (os.getenv("EARTHDATA_USERNAME") and os.getenv("EARTHDATA_PASSWORD"))
            or _netrc_credentials()
        ):
            pytest.skip("needs NASA Earthdata credentials")
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

    def test_find_granules_filters_by_center_time(self, fetch):
        auth = _EarthdataAuth()
        result = SMAPL4Reaper(dates=DATES)._find_granules(auth)

        assert [t for t, _ in result] == [
            pd.Timestamp("2026-09-28 01:30", tz="UTC"),
            pd.Timestamp("2026-09-28 04:30", tz="UTC"),
        ]
        assert result[0][1] == f"{OPENDAP}/00.h5"
        assert fetch.call_args.kwargs["request_kwargs"] == [
            {
                "params": {
                    "short_name": "SPL4SMGP",
                    "version": "008",
                    "temporal": "2026-09-28T00:00:00Z,2026-09-28T06:00:00Z",
                    "page_size": 2000,
                },
                "middlewares": (auth,),
            }
        ]

    def test_find_granules_latest(self, fetch):
        reaper = SMAPL4Reaper(dates="latest")
        reaper.start_date = pd.Timestamp("2026-09-27", tz="UTC")

        assert [t for t, _ in reaper._find_granules(_EarthdataAuth())] == [
            pd.Timestamp("2026-09-28 04:30", tz="UTC")
        ]

    def test_find_granules_splits_long_ranges(self, fetch):
        result = SMAPL4Reaper(dates=("2025-06-01", DATES[1]))._find_granules(_EarthdataAuth())

        temporal = [kw["params"]["temporal"] for kw in fetch.call_args.kwargs["request_kwargs"]]
        assert temporal == [
            "2025-06-01T00:00:00Z,2025-12-18T00:00:00Z",
            "2025-12-18T00:00:00Z,2026-07-06T00:00:00Z",
            "2026-07-06T00:00:00Z,2026-09-28T06:00:00Z",
        ]
        # The same granules come back from every window; duplicates are dropped.
        assert len(result) == 3

    def test_find_granules_none(self, mocker, credentials):
        mocker.patch(
            "cosecha.reaping.smap.tiny_retriever.fetch", return_value=[{"hits": 0, "items": []}]
        )
        with pytest.raises(DataNotFoundError, match="No SPL4SMGP granules"):
            SMAPL4Reaper(dates=DATES)._find_granules(_EarthdataAuth())

    def test_find_granules_truncated_page(self, mocker, credentials):
        mocker.patch(
            "cosecha.reaping.smap.tiny_retriever.fetch",
            return_value=[{"hits": 2001, "items": GRANULES}],
        )
        with pytest.raises(APIError, match="CMR returned 3 of 2001"):
            SMAPL4Reaper(dates=DATES)._find_granules(_EarthdataAuth())

    def test_find_granules_search_error(self, mocker, credentials):
        mocker.patch(
            "cosecha.reaping.smap.tiny_retriever.fetch", side_effect=RuntimeError("CMR down")
        )
        with pytest.raises(APIError, match="Could not search NASA CMR"):
            SMAPL4Reaper(dates=DATES)._find_granules(_EarthdataAuth())

    @pytest.mark.parametrize(
        "opendap", ["http://opendap.earthdata.nasa.gov/c", "https://evil.example.com/c"]
    )
    def test_find_granules_rejects_unexpected_host(self, mocker, credentials, opendap):
        mocker.patch(
            "cosecha.reaping.smap.tiny_retriever.fetch",
            return_value=[{"hits": 1, "items": [_granule("2026-09-28T00:00", opendap)]}],
        )
        with pytest.raises(APIError, match="Unexpected SPL4SMGP OPeNDAP URL"):
            SMAPL4Reaper(dates=DATES)._find_granules(_EarthdataAuth())

    def test_reap_with_spatial_subset(self, fetch):
        reaper = SMAPL4Reaper(
            dates=DATES,
            variables=["sm_surface", "sm_rootzone"],
            transformations={**BBOX, "variable_rename": {"sm_surface": "sm"}},
            max_workers=2,
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

        # CMR search, one call per coordinate (the first logs in), one call for all granules.
        assert [len(c.args[0]) for c in fetch.call_args_list] == [1, 1, 1, 2]
        call = fetch.call_args
        assert call.args[0] == [f"{OPENDAP}/00.h5.dap.nc4", f"{OPENDAP}/03.h5.dap.nc4"]
        assert call.kwargs["limit_per_host"] == 2
        ce = (
            "/Geophysical_Data/sm_surface[1:1:3][2:1:4];/Geophysical_Data/sm_rootzone[1:1:3][2:1:4]"
        )
        auth = call.kwargs["request_kwargs"][0]["middlewares"][0]
        assert isinstance(auth, _EarthdataAuth)
        for kwargs in call.kwargs["request_kwargs"]:
            assert kwargs == {"params": {"dap4.ce": ce}, "middlewares": (auth,)}

    def test_reap_with_netrc(self, fetch, monkeypatch, netrc_file):
        for var in CREDENTIAL_VARS:
            monkeypatch.delenv(var, raising=False)
        ds = SMAPL4Reaper(dates=DATES, variables=["sm_surface"], transformations=BBOX).reap()

        assert dict(ds.sizes) == {"time": 2, "latitude": 3, "longitude": 3}

    def test_reap_global(self, fetch):
        ds = SMAPL4Reaper(dates=DATES, variables=["sm_surface"]).reap()

        assert dict(ds.sizes) == {"time": 2, "latitude": 5, "longitude": 6}
        assert bool((ds.latitude.diff("latitude") > 0).all())

    def test_subset_without_cells(self, fetch):
        reaper = SMAPL4Reaper(
            dates=DATES, transformations={"spatial_subset": {"lat_bounds": (33.2, 33.8)}}
        )
        with pytest.raises(DataNotFoundError, match="contains no SMAP cells"):
            reaper.reap()

    def test_reap_wraps_request_errors(self, fetch):
        fetch.side_effect = [[{"hits": 3, "items": GRANULES}], _service_error(500)]
        with pytest.raises(ReaperError, match="SMAP L4 reaping failed") as exc:
            SMAPL4Reaper(dates=DATES).reap()
        assert not isinstance(exc.value, APIError)

    @pytest.mark.parametrize("status", [401, 403])
    def test_reap_rejected_credentials(self, fetch, status):
        fetch.side_effect = [[{"hits": 3, "items": GRANULES}], _service_error(status)]
        with pytest.raises(APIError, match=f"rejected the credentials \\(HTTP {status}\\)"):
            SMAPL4Reaper(dates=DATES).reap()

    def test_reap_fetches_granules_in_batches_of_max_workers(self, fetch):
        SMAPL4Reaper(dates=DATES, max_workers=1).reap()

        # CMR search, two coordinate requests, then one granule per call.
        assert [len(c.args[0]) for c in fetch.call_args_list] == [1, 1, 1, 1, 1]
        assert [c.args[0] for c in fetch.call_args_list[3:]] == [
            [f"{OPENDAP}/00.h5.dap.nc4"],
            [f"{OPENDAP}/03.h5.dap.nc4"],
        ]

    def test_reap_timeout(self, fetch):
        err = ServiceError("", OPENDAP)
        err.__cause__ = TimeoutError()
        fetch.side_effect = [[{"hits": 3, "items": GRANULES}], err]
        with pytest.raises(ReaperError, match="timed out after 120 s"):
            SMAPL4Reaper(dates=DATES).reap()

    def test_reap_session_expired_during_batch(self, fetch):
        def relogin(urls, return_type, *, request_kwargs, **kwargs):
            if len(urls) > 1:
                request_kwargs[0]["middlewares"][0].logins += 1
            return _fake_fetch(urls, return_type, request_kwargs=request_kwargs, **kwargs)

        fetch.side_effect = relogin
        with pytest.raises(APIError, match="session expired while downloading"):
            SMAPL4Reaper(dates=DATES).reap()

    def test_reap_without_credentials(self, fetch, monkeypatch):
        for var in CREDENTIAL_VARS:
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(APIError, match="credentials not found"):
            SMAPL4Reaper(dates=DATES).reap()
        fetch.assert_not_called()

    def test_network_test_skips_without_credentials(self, credentials, monkeypatch):
        for var in CREDENTIAL_VARS:
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(pytest.skip.Exception, match="needs NASA Earthdata credentials"):
            TestSMAPL4ReaperNetwork().test_reap_network()

    def test_network_test_accepts_netrc(self, credentials, monkeypatch, netrc_file, mocker):
        for var in CREDENTIAL_VARS:
            monkeypatch.delenv(var, raising=False)
        ds = xr.Dataset(
            {"sm_surface": (("time", "latitude", "longitude"), np.zeros((2, 1, 1)))},
            coords={"latitude": [33.0], "longitude": [-97.0]},
        )
        reap = mocker.patch.object(SMAPL4Reaper, "reap", return_value=ds)

        TestSMAPL4ReaperNetwork().test_reap_network()

        reap.assert_called_once()


def _request_headers(
    url: str, auth: _EarthdataAuth | None = None, headers: dict | None = None, **cookies: str
) -> dict[str, str]:
    """Run the auth middleware on a real aiohttp request and return its headers."""

    async def handler(req):
        morsels = {k: SimpleNamespace(value=v) for k, v in cookies.items()}
        return SimpleNamespace(headers=dict(req.headers), cookies=morsels)

    async def run():
        request = ClientRequest("GET", URL(url), headers=headers)
        return await (auth or _EarthdataAuth())(request, handler)

    return asyncio.run(run()).headers


def _authorization(
    url: str, auth: _EarthdataAuth | None = None, headers: dict | None = None, **cookies: str
) -> str | None:
    return _request_headers(url, auth, headers, **cookies).get("Authorization")


class TestAuthMiddleware:
    """Test which hosts receive Earthdata credentials."""

    @pytest.fixture(autouse=True)
    def no_credentials(self, monkeypatch, isolated_netrc):
        for var in CREDENTIAL_VARS:
            monkeypatch.delenv(var, raising=False)

    def test_username_password_only_sent_to_earthdata_login(self, monkeypatch):
        monkeypatch.setenv("EARTHDATA_USERNAME", "user")
        monkeypatch.setenv("EARTHDATA_PASSWORD", "p!w:)")
        basic = "Basic " + base64.b64encode(b"user:p!w:)").decode()

        assert _authorization("https://urs.earthdata.nasa.gov/oauth/authorize") == basic
        assert _authorization("http://urs.earthdata.nasa.gov/oauth/authorize") is None
        assert _authorization(f"{OPENDAP}/00.h5.dap.nc4") is None
        assert _authorization("https://example.com/") is None

    def test_drops_other_authorization_headers(self, monkeypatch):
        """A token overrides headers aiohttp takes from ~/.netrc."""
        monkeypatch.setenv("EARTHDATA_TOKEN", "abc")
        netrc = {"Authorization": "Basic bmV0cmM6anVuaw=="}

        assert _authorization(_CMR_URL, headers=netrc) is None
        assert _authorization("https://urs.earthdata.nasa.gov/", headers=netrc) is None
        assert _authorization(f"{OPENDAP}/00.h5.dap.nc4", headers=netrc) == "Bearer abc"

    def test_token_only_sent_to_opendap(self, monkeypatch):
        monkeypatch.setenv("EARTHDATA_TOKEN", "abc")

        assert _authorization(f"{OPENDAP}/00.h5.dap.nc4") == "Bearer abc"
        assert _authorization("https://urs.earthdata.nasa.gov/oauth/authorize") is None
        assert _authorization("https://example.com/") is None

    def test_username_password_preferred_over_token(self, monkeypatch):
        monkeypatch.setenv("EARTHDATA_TOKEN", "expired")
        monkeypatch.setenv("EARTHDATA_USERNAME", "user")
        monkeypatch.setenv("EARTHDATA_PASSWORD", "pw")

        assert _authorization(f"{OPENDAP}/00.h5.dap.nc4") is None
        assert _authorization("https://urs.earthdata.nasa.gov/").startswith("Basic ")

    def test_username_without_password(self, monkeypatch):
        monkeypatch.setenv("EARTHDATA_USERNAME", "user")
        with pytest.raises(APIError, match="credentials not found"):
            _EarthdataAuth()

    @pytest.mark.parametrize("default_path", [False, True])
    def test_netrc_credentials_only_sent_to_earthdata_login(
        self, monkeypatch, netrc_file, default_path
    ):
        if default_path:
            monkeypatch.delenv("NETRC")
            monkeypatch.setenv("HOME", str(netrc_file.parent))
            monkeypatch.setenv("USERPROFILE", str(netrc_file.parent))
        auth = _EarthdataAuth()
        basic = "Basic " + base64.b64encode(b"netrc-user:netrc-pw").decode()

        assert _authorization("https://urs.earthdata.nasa.gov/", auth) == basic
        assert _authorization("http://urs.earthdata.nasa.gov/", auth) is None
        assert _authorization(OPENDAP, auth) is None
        assert _authorization("https://example.com/", auth) is None

    def test_environment_credentials_preferred_over_netrc(self, monkeypatch, netrc_file):
        monkeypatch.setenv("EARTHDATA_USERNAME", "env-user")
        monkeypatch.setenv("EARTHDATA_PASSWORD", "env-pw")
        basic = "Basic " + base64.b64encode(b"env-user:env-pw").decode()

        assert _authorization("https://urs.earthdata.nasa.gov/") == basic

    def test_token_preferred_over_netrc(self, monkeypatch, netrc_file):
        monkeypatch.setenv("EARTHDATA_TOKEN", "abc")

        assert _authorization("https://urs.earthdata.nasa.gov/") is None
        assert _authorization(OPENDAP) == "Bearer abc"

    def test_incomplete_environment_credentials_fall_back_to_netrc(self, monkeypatch, netrc_file):
        monkeypatch.setenv("EARTHDATA_USERNAME", "env-user")
        basic = "Basic " + base64.b64encode(b"netrc-user:netrc-pw").decode()

        assert _authorization("https://urs.earthdata.nasa.gov/") == basic

    @pytest.mark.parametrize(
        "content",
        [
            "machine example.com login other password other-pw\n",
            "machine urs.earthdata.nasa.gov login user\n",
        ],
    )
    def test_netrc_without_earthdata_credentials(self, netrc_file, content):
        netrc_file.write_text(content)
        with pytest.raises(APIError, match="credentials not found"):
            _EarthdataAuth()

    def test_invalid_netrc(self, netrc_file):
        netrc_file.write_text("machine urs.earthdata.nasa.gov unexpected\n")
        with pytest.raises(APIError, match="netrc"):
            _EarthdataAuth()

    def test_unreadable_netrc(self, mocker, netrc_file):
        mocker.patch("cosecha.reaping.smap.netrc.netrc", side_effect=PermissionError())
        with pytest.raises(APIError, match="netrc"):
            _EarthdataAuth()

    def test_saved_cookies_only_sent_to_https_opendap(self, monkeypatch):
        monkeypatch.setenv("EARTHDATA_USERNAME", "user")
        monkeypatch.setenv("EARTHDATA_PASSWORD", "pw")
        auth = _EarthdataAuth()
        auth.cookies["JSESSIONID"] = "session"

        assert _request_headers(OPENDAP, auth)["Cookie"] == "JSESSIONID=session"
        for url in (
            "http://opendap.earthdata.nasa.gov/",
            "https://urs.earthdata.nasa.gov/",
            "https://example.com/",
        ):
            assert "Cookie" not in _request_headers(url, auth)

    def test_counts_logins_and_keeps_opendap_cookies(self, monkeypatch):
        monkeypatch.setenv("EARTHDATA_USERNAME", "user")
        monkeypatch.setenv("EARTHDATA_PASSWORD", "pw")
        auth = _EarthdataAuth()

        _authorization(f"{OPENDAP}/00.h5.dap.nc4", auth, JSESSIONID="a")
        _authorization("https://urs.earthdata.nasa.gov/oauth/authorize", auth, urs="x")
        _authorization(f"{OPENDAP}/login/urs?code=1", auth, JSESSIONID="b")

        assert auth.logins == 1
        assert auth.cookies == {"JSESSIONID": "b"}
