"""SMAP L4 reaper for gridded surface and root-zone soil moisture.

Reaps the NASA SMAP L4 Global 3-hourly 9 km soil moisture product (SPL4SMGP). Granules
are found through NASA CMR with earthaccess and subset server-side by the Earthdata
Cloud OPeNDAP service, so only the requested cells and variables are downloaded.
"""

from __future__ import annotations

import io
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd
import requests
import xarray as xr
from requests.adapters import HTTPAdapter
from requests.utils import get_netrc_auth
from urllib3.util import Retry

from cosecha._logging import logger
from cosecha._utils import apply_gridded_transformations, parse_date_range, wrap_errors
from cosecha.exceptions import APIError, DataNotFoundError, ReaperError, TransformationError
from cosecha.reaping.base import GriddedReaper

__all__ = ["SMAPL4Reaper"]

SHORT_NAME = "SPL4SMGP"
VERSION = "008"
DEFAULT_VARIABLES = (
    "sm_surface",
    "sm_rootzone",
    "sm_profile",
    "sm_surface_wetness",
    "sm_rootzone_wetness",
    "sm_profile_wetness",
)
# All Geophysical_Data variables in SPL4SMGP v008.
VALID_VARIABLES = frozenset(
    {
        "baseflow_flux",
        "depth_to_water_table_from_surface",
        "free_surface_water_on_peat_flux",
        "heat_flux_ground",
        "heat_flux_latent",
        "heat_flux_sensible",
        "height_lowatmmodlay",
        "land_evapotranspiration_flux",
        "land_fraction_saturated",
        "land_fraction_snow_covered",
        "land_fraction_unsaturated",
        "land_fraction_wilting",
        "leaf_area_index",
        "mwrtm_vegopacity",
        "net_downward_longwave_flux",
        "net_downward_shortwave_flux",
        "overland_runoff_flux",
        "precipitation_total_surface_flux",
        "radiation_longwave_absorbed_flux",
        "radiation_shortwave_downward_flux",
        "sm_profile",
        "sm_profile_pctl",
        "sm_profile_wetness",
        "sm_rootzone",
        "sm_rootzone_pctl",
        "sm_rootzone_wetness",
        "sm_surface",
        "sm_surface_wetness",
        "snow_depth",
        "snow_mass",
        "snow_melt_flux",
        "snowfall_surface_flux",
        "soil_temp_layer1",
        "soil_temp_layer2",
        "soil_temp_layer3",
        "soil_temp_layer4",
        "soil_temp_layer5",
        "soil_temp_layer6",
        "soil_water_infiltration_flux",
        "specific_humidity_lowatmmodlay",
        "surface_pressure",
        "surface_temp",
        "temp_lowatmmodlay",
        "vegetation_greenness_fraction",
        "windspeed_lowatmmodlay",
    }
)
# Global 9 km EASE-Grid 2.0: latitude decreases with row, longitude increases with column.
_NROWS, _NCOLS = 1624, 3856
# SMAP L4 latency is a few days; "latest" searches this far back.
_LATEST_LOOKBACK = timedelta(days=14)
_EDL_HOST = "urs.earthdata.nasa.gov"


class SMAPL4Reaper(GriddedReaper):
    """Reaper for NASA SMAP L4 3-hourly 9 km soil moisture (SPL4SMGP)."""

    def __init__(
        self,
        dates: Literal["latest"] | tuple[str, str],
        variables: list[str] | tuple[str, ...] = DEFAULT_VARIABLES,
        transformations: dict[str, Any] | None = None,
        max_workers: int = 4,
        timeout: int = 120,
    ) -> None:
        """Initialize SMAPL4Reaper.

        Each time step is the 3-hour granule centered on that time (e.g. 01:30 covers
        00:00-03:00 UTC). Requires a free NASA Earthdata Login, read from the
        ``EARTHDATA_TOKEN`` (or ``EARTHDATA_USERNAME``/``EARTHDATA_PASSWORD``) environment
        variables, or from ``~/.netrc``.

        Parameters
        ----------
        dates : Literal["latest"] | tuple[str, str]
            "latest" for the most recent granule, or a (start_time, end_time) tuple.
            Granules centered within [start_time, end_time] are returned. A date-only end
            means midnight, so ("2026-09-28", "2026-10-03") ends with the 2026-10-02 22:30
            granule; use "2026-10-03 23:59" to include all of Oct 3.
        variables : list[str] | tuple[str, ...], optional
            Variables from the ``Geophysical_Data`` group (see ``VALID_VARIABLES``).
            Defaults to surface, root-zone and profile soil moisture (m3 m-3) and wetness
            (dimensionless).
        transformations : dict[str, Any] | None, optional
            Optional transformations. ``spatial_subset`` ({"lat_bounds": (min, max),
            "lon_bounds": (min, max)}, longitude in -180 to 180) is applied server-side, so
            only cells within the bounds are downloaded; without it the global grid is
            fetched (~25 MB per variable per time step).
        max_workers : int, optional
            Number of granules fetched concurrently, by default 4.
        timeout : int, optional
            Request timeout in seconds, by default 120.

        Raises
        ------
        ImportError
            If earthaccess is not installed.
        DateRangeError
            If dates cannot be parsed or start is after end.
        ReaperError
            If variables are unknown or max_workers/timeout are not positive.
        TransformationError
            If spatial_subset has unknown keys or invalid bounds.

        Examples
        --------
        >>> reaper = SMAPL4Reaper(
        ...     dates=("2026-09-28", "2026-10-03"),
        ...     variables=["sm_surface", "sm_rootzone"],
        ...     transformations={
        ...         "spatial_subset": {"lat_bounds": (31.5, 34), "lon_bounds": (-98.5, -95.5)},
        ...     },
        ... )
        """
        super().__init__()
        try:
            import earthaccess  # noqa: F401,PLC0415
        except ImportError:
            raise ImportError(
                "earthaccess is not installed. Install with: pip install 'cosecha[earthdata]'"
            ) from None

        if dates != "latest" and not (isinstance(dates, (tuple, list)) and len(dates) == 2):
            raise ValueError(
                'dates must be "latest" or a sequence of length 2, e.g. ("2026-09-28", "2026-10-03")'
            )

        self.is_latest = dates == "latest"
        if self.is_latest:
            self.end_date = pd.Timestamp.now(tz="UTC")
            self.start_date = self.end_date - _LATEST_LOOKBACK
        else:
            self.start_date, self.end_date = parse_date_range(*dates)

        self.variables = list(variables)
        self.transformations = dict(transformations or {})
        self.bounds: dict[str, tuple[float, float]] = self.transformations.pop("spatial_subset", {})
        self.max_workers = max_workers
        self.timeout = timeout

        self._validate_params()
        logger.debug(f"Initialized {self.__class__.__name__}: dates={dates}, variables={variables}")

    def _validate_params(self) -> None:
        """Validate initialization parameters."""
        if not self.variables:
            raise ReaperError("variables must contain at least one variable name.")
        if unknown := sorted(set(self.variables) - VALID_VARIABLES):
            raise ReaperError(
                f"Unknown {SHORT_NAME} variables {unknown}. Valid: {sorted(VALID_VARIABLES)}"
            )
        if self.max_workers < 1 or self.timeout <= 0:
            raise ReaperError(
                f"max_workers and timeout must be positive, got {self.max_workers}, {self.timeout}"
            )

        if unknown := sorted(set(self.bounds) - {"lat_bounds", "lon_bounds"}):
            raise TransformationError(
                f"Unknown spatial_subset keys {unknown}; expected 'lat_bounds' and/or 'lon_bounds'."
            )
        for key, limit in (("lat_bounds", 90), ("lon_bounds", 180)):
            bounds = self.bounds.get(key)
            if bounds is not None and not (
                len(bounds) == 2 and -limit <= bounds[0] <= bounds[1] <= limit
            ):
                raise TransformationError(
                    f"{key} must be (min, max) with -{limit} <= min <= max <= {limit}, got "
                    f"{bounds}. Ranges crossing the antimeridian are not supported."
                )

    def _login(self) -> requests.Session:
        """Return a retrying session authorized for NASA Earthdata.

        Username/password are sent to Earthdata Login on OPeNDAP's redirect rather than
        exchanged for a token, as the token API rejects some valid accounts.
        """
        from earthaccess.auth import SessionWithHeaderRedirection  # noqa: PLC0415

        token = os.getenv("EARTHDATA_TOKEN")
        username, password = os.getenv("EARTHDATA_USERNAME"), os.getenv("EARTHDATA_PASSWORD")
        if token:
            session = requests.Session()
            session.headers["Authorization"] = f"Bearer {token}"
        elif (
            creds := (username, password)
            if username and password
            else get_netrc_auth(f"https://{_EDL_HOST}")
        ):
            session = SessionWithHeaderRedirection(_EDL_HOST, creds)
        else:
            raise APIError(
                "NASA Earthdata credentials not found. Set EARTHDATA_TOKEN or "
                "EARTHDATA_USERNAME and EARTHDATA_PASSWORD, or add "
                f"{_EDL_HOST} to ~/.netrc (register at https://{_EDL_HOST})."
            )

        retry = Retry(total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))
        session.mount("https://", HTTPAdapter(max_retries=retry))
        return session

    def _find_granules(self) -> list[tuple[pd.Timestamp, str]]:
        """Search CMR and return (center time, OPeNDAP URL) pairs sorted by time."""
        import earthaccess  # noqa: PLC0415

        with wrap_errors(APIError, f"Could not search NASA CMR for {SHORT_NAME}"):
            results = earthaccess.search_data(
                short_name=SHORT_NAME,
                version=VERSION,
                temporal=(self.start_date.isoformat(), self.end_date.isoformat()),
            )

        granules: dict[pd.Timestamp, str] = {}
        for granule in results:
            umm = granule["umm"]
            url = next(
                (u["URL"] for u in umm["RelatedUrls"] if u.get("Subtype") == "OPENDAP DATA"),
                None,
            )
            span = umm["TemporalExtent"]["RangeDateTime"]
            begin, end = pd.to_datetime(
                [span["BeginningDateTime"], span["EndingDateTime"]], utc=True
            )
            center = begin + (end - begin) / 2
            # Duplicate granules (e.g. reprocessed copies) share a center time; last wins.
            if url and self.start_date <= center <= self.end_date:
                granules[center] = url

        if not granules:
            raise DataNotFoundError(
                f"No {SHORT_NAME} granules found between {self.start_date} and {self.end_date}."
            )
        ordered = sorted(granules.items())
        return ordered[-1:] if self.is_latest else ordered

    def _open(
        self, session: requests.Session, url: str, constraint: str, group: str | None = None
    ) -> xr.Dataset:
        """Load a DAP4-constrained NetCDF-4 subset from OPeNDAP into memory."""
        response = session.get(
            f"{url}.dap.nc4", params={"dap4.ce": constraint}, timeout=self.timeout
        )
        response.raise_for_status()
        return xr.load_dataset(
            io.BytesIO(response.content), engine="h5netcdf", group=group, phony_dims="sort"
        )

    def _fetch_data(self) -> xr.Dataset:
        """Fetch the requested SMAP L4 variables, time steps and cells via OPeNDAP.

        Returns
        -------
        xr.Dataset
            Dataset with dims (time, latitude, longitude), latitude ascending.
        """
        granules = self._find_granules()
        session = self._login()
        logger.info(f"Fetching {len(granules)} {SHORT_NAME} granules via OPeNDAP")

        # Separate requests: a lat column and lon row can't share (y, x) dims in one file.
        url = granules[0][1]
        lat = self._open(session, url, f"/cell_lat[0:1:{_NROWS - 1}][0]")["cell_lat"].values.ravel()
        lon = self._open(session, url, f"/cell_lon[0][0:1:{_NCOLS - 1}]")["cell_lon"].values.ravel()
        rows = _index_slice(lat, self.bounds.get("lat_bounds"), "lat_bounds")
        cols = _index_slice(lon, self.bounds.get("lon_bounds"), "lon_bounds")
        if not self.bounds:
            logger.warning("No spatial_subset given; fetching the global SMAP L4 grid.")

        hyperslab = f"[{rows.start}:1:{rows.stop - 1}][{cols.start}:1:{cols.stop - 1}]"
        constraint = ";".join(f"/Geophysical_Data/{v}{hyperslab}" for v in self.variables)

        def fetch(granule: tuple[pd.Timestamp, str]) -> xr.Dataset:
            time, url = granule
            logger.debug(f"Fetching {SHORT_NAME} granule {time}")
            ds = self._open(session, url, constraint, group="Geophysical_Data")[self.variables]
            ds = ds.drop_vars(list(ds.coords))
            # All Geophysical_Data variables share the same (y, x) grid.
            dims = dict(zip(ds[self.variables[0]].dims, ("latitude", "longitude"), strict=True))
            return ds.rename_dims(dims).expand_dims(time=[time.tz_convert(None)])

        with ThreadPoolExecutor(self.max_workers) as pool:
            ds = xr.concat(list(pool.map(fetch, granules)), dim="time")

        ds = ds.assign_coords(latitude=lat[rows], longitude=lon[cols])
        # Flip so latitude ascends, matching the other gridded reapers.
        return ds.isel(latitude=slice(None, None, -1))

    def _reap(self) -> xr.Dataset:
        """Fetch and return SMAP L4 gridded data.

        Returns
        -------
        xr.Dataset
            SMAP L4 gridded dataset.

        Raises
        ------
        DataNotFoundError
            If no granules exist for the requested dates or the subset has no cells.
        APIError
            If CMR search or Earthdata login fails.
        ReaperError
            If OPeNDAP requests or processing fail.
        """
        period = "latest" if self.is_latest else f"{self.start_date} to {self.end_date}"
        logger.info(f"Reaping SMAP L4 data for {period}")

        with wrap_errors(ReaperError, "SMAP L4 reaping failed", ReaperError):
            ds = apply_gridded_transformations(self._fetch_data(), self.transformations)
            logger.info(f"Successfully reaped SMAP L4 data: {ds.sizes['time']} time steps")
            return ds


def _index_slice(coord: np.ndarray, bounds: tuple[float, float] | None, label: str) -> slice:
    """Return the index range of ``coord`` values within ``bounds``."""
    if bounds is None:
        return slice(0, coord.size)
    idx = np.flatnonzero((coord >= bounds[0]) & (coord <= bounds[1]))
    if idx.size == 0:
        raise DataNotFoundError(f"spatial_subset {label}={bounds} contains no SMAP cells.")
    return slice(int(idx[0]), int(idx[-1]) + 1)
