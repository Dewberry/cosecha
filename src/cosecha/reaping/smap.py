"""SMAP L4 reaper for gridded surface and root-zone soil moisture.

Reaps the NASA SMAP L4 Global 3-hourly 9 km soil moisture product (SPL4SMGP). Granules
are found through NASA CMR and subset server-side by the Earthdata Cloud OPeNDAP
service, so only the requested cells and variables are downloaded.
"""

from __future__ import annotations

import base64
import io
import itertools
import netrc
import os
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import tiny_retriever
import xarray as xr
from aiohttp import ClientResponseError, hdrs
from tiny_retriever.exceptions import ServiceError

from cosecha._logging import logger
from cosecha._utils import apply_gridded_transformations, parse_date_range, wrap_errors
from cosecha.exceptions import APIError, DataNotFoundError, ReaperError, TransformationError
from cosecha.reaping.base import GriddedReaper

if TYPE_CHECKING:
    from aiohttp import ClientHandlerType, ClientRequest, ClientResponse

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
_OPENDAP_HOST = "opendap.earthdata.nasa.gov"
_CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
# CMR returns at most 2000 granules per page; 8 granules/day keeps a 200-day window under it.
_CMR_PAGE_SIZE = 2000
_CMR_WINDOW = timedelta(days=200)


def _netrc_credentials() -> tuple[str, str] | None:
    """Read Earthdata Login credentials from NETRC or the default ~/.netrc."""
    try:
        entry = netrc.netrc(os.getenv("NETRC") or None).authenticators(_EDL_HOST)
    except FileNotFoundError:
        return None
    except (OSError, netrc.NetrcParseError) as e:
        raise APIError(
            "Could not read Earthdata credentials from netrc; check its syntax and permissions."
        ) from e
    if entry:
        login, account, password = entry
        username = login or account
        if username and password:
            return username, password
    return None


class _EarthdataAuth:
    """aiohttp middleware that adds Earthdata credentials on NASA hosts only.

    Username/password are sent to Earthdata Login on OPeNDAP's OAuth redirect rather than
    exchanged for a token, as the EDL token API rejects some valid accounts. A token is
    sent to OPeNDAP directly. Environment username/password take precedence over a token,
    followed by Earthdata Login credentials from ``NETRC`` or ``~/.netrc``.

    Any other ``Authorization`` header, e.g. from a ``~/.netrc`` entry that aiohttp picks up,
    is removed, so only the selected credentials reach their intended host.

    OPeNDAP stores the URL to return to after login in its session, so concurrent logins
    sharing a cookie jar get sent back to each other's URLs. Log in with a single request
    first; later requests reuse the session cookies captured here and skip the redirect.
    """

    def __init__(self) -> None:
        username, password = os.getenv("EARTHDATA_USERNAME"), os.getenv("EARTHDATA_PASSWORD")
        token = os.getenv("EARTHDATA_TOKEN")
        if not (username and password) and not token:
            username, password = _netrc_credentials() or (None, None)
        if username and password:
            self._host = _EDL_HOST
            self._value = "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()
        elif token:
            self._host, self._value = _OPENDAP_HOST, f"Bearer {token}"
        else:
            raise APIError(
                "NASA Earthdata credentials not found. Set EARTHDATA_USERNAME and "
                "EARTHDATA_PASSWORD (or EARTHDATA_TOKEN), or add an "
                f"{_EDL_HOST} entry to NETRC or ~/.netrc; register at "
                f"https://{_EDL_HOST}."
            )
        self.cookies: dict[str, str] = {}
        self.logins = 0

    async def __call__(self, request: ClientRequest, handler: ClientHandlerType) -> ClientResponse:
        if request.url.scheme == "https" and request.url.host == self._host:
            request.headers[hdrs.AUTHORIZATION] = self._value
            self.logins += self._host == _EDL_HOST
        else:
            request.headers.pop(hdrs.AUTHORIZATION, None)
        if request.url.scheme == "https" and request.url.host == _OPENDAP_HOST:
            request.update_cookies(self.cookies)
        response = await handler(request)
        if request.url.scheme == "https" and request.url.host == _OPENDAP_HOST:
            self.cookies.update({name: morsel.value for name, morsel in response.cookies.items()})
        return response


def _cmr_time(ts: pd.Timestamp) -> str:
    return f"{ts:%Y-%m-%dT%H:%M:%SZ}"


def _load(blob: bytes, group: str | None = None) -> xr.Dataset:
    """Load an OPeNDAP NetCDF-4 response into memory."""
    return xr.load_dataset(io.BytesIO(blob), engine="h5netcdf", group=group, phony_dims="sort")


def _index_slice(coord: np.ndarray, bounds: tuple[float, float] | None, label: str) -> slice:
    """Return the index range of ``coord`` values within ``bounds``."""
    if bounds is None:
        return slice(0, coord.size)
    idx = np.flatnonzero((coord >= bounds[0]) & (coord <= bounds[1]))
    if idx.size == 0:
        raise DataNotFoundError(f"spatial_subset {label}={bounds} contains no SMAP cells.")
    return slice(int(idx[0]), int(idx[-1]) + 1)


class SMAPL4Reaper(GriddedReaper):
    """Reaper for NASA SMAP L4 3-hourly 9 km soil moisture (SPL4SMGP)."""

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
        ``EARTHDATA_USERNAME`` and ``EARTHDATA_PASSWORD`` environment variables, or an
        Earthdata token from ``EARTHDATA_TOKEN`` (tokens expire after 60 days). If neither
        is set, reads the ``urs.earthdata.nasa.gov`` entry from ``NETRC`` or ``~/.netrc``.

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

    def _find_granules(self, auth: _EarthdataAuth) -> list[tuple[pd.Timestamp, str]]:
        """Search CMR and return (center time, OPeNDAP URL) pairs sorted by time."""
        edges = pd.date_range(self.start_date, self.end_date, freq=_CMR_WINDOW)
        edges = edges.append(pd.DatetimeIndex([self.end_date])).unique()
        windows = list(itertools.pairwise(edges)) or [(self.start_date, self.end_date)]
        kwargs = [
            {
                "params": {
                    "short_name": SHORT_NAME,
                    "version": VERSION,
                    "temporal": f"{_cmr_time(start)},{_cmr_time(end)}",
                    "page_size": _CMR_PAGE_SIZE,
                },
                "middlewares": (auth,),
            }
            for start, end in windows
        ]
        with wrap_errors(APIError, f"Could not search NASA CMR for {SHORT_NAME}"):
            pages = tiny_retriever.fetch(
                [_CMR_URL] * len(windows), "json", request_kwargs=kwargs, timeout=self.timeout
            )

        granules: dict[pd.Timestamp, str] = {}
        for page in pages:
            if page["hits"] > len(page["items"]):
                raise APIError(
                    f"CMR returned {len(page['items'])} of {page['hits']} {SHORT_NAME} granules."
                )
            for item in page["items"]:
                umm = item["umm"]
                url = next(
                    (u["URL"] for u in umm["RelatedUrls"] if u.get("Subtype") == "OPENDAP DATA"),
                    None,
                )
                span = umm["TemporalExtent"]["RangeDateTime"]
                begin, end = pd.to_datetime(
                    [span["BeginningDateTime"], span["EndingDateTime"]], utc=True
                )
                center = begin + (end - begin) / 2
                if not url or not self.start_date <= center <= self.end_date:
                    continue
                parts = urlsplit(url)
                if parts.scheme != "https" or parts.hostname != _OPENDAP_HOST:
                    raise APIError(f"Unexpected {SHORT_NAME} OPeNDAP URL from CMR: {url}")
                # Duplicate granules (e.g. reprocessed copies or window overlaps) share a
                # center time; last wins.
                granules[center] = url

        if not granules:
            raise DataNotFoundError(
                f"No {SHORT_NAME} granules found between {self.start_date} and {self.end_date}."
            )
        ordered = sorted(granules.items())
        return ordered[-1:] if self.is_latest else ordered

    def _fetch_dap(
        self, urls: list[str], constraints: list[str], auth: _EarthdataAuth
    ) -> list[bytes]:
        """Download DAP4-constrained NetCDF-4 subsets from OPeNDAP concurrently."""
        kwargs = [{"params": {"dap4.ce": ce}, "middlewares": (auth,)} for ce in constraints]
        try:
            return tiny_retriever.fetch(
                [f"{url}.dap.nc4" for url in urls],
                "binary",
                request_kwargs=kwargs,
                limit_per_host=self.max_workers,
                timeout=self.timeout,
            )
        except ServiceError as e:
            cause = e.__cause__
            if isinstance(cause, ClientResponseError) and cause.status in (401, 403):
                raise APIError(
                    f"NASA Earthdata rejected the credentials (HTTP {cause.status}). Check "
                    "EARTHDATA_USERNAME and EARTHDATA_PASSWORD or the Earthdata netrc entry, "
                    "or renew EARTHDATA_TOKEN."
                ) from e
            if isinstance(cause, TimeoutError):
                raise ReaperError(
                    f"OPeNDAP request timed out after {self.timeout} s; try a larger timeout."
                ) from e
            raise

    def _fetch_data(self) -> xr.Dataset:
        """Fetch the requested SMAP L4 variables, time steps and cells via OPeNDAP.

        Returns
        -------
        xr.Dataset
            Dataset with dims (time, latitude, longitude), latitude ascending.
        """
        auth = _EarthdataAuth()
        granules = self._find_granules(auth)
        logger.info(f"Fetching {len(granules)} {SHORT_NAME} granules via OPeNDAP")

        # One request at a time, so the first one logs in before any concurrent requests.
        # Separate requests: a lat column and lon row can't share (y, x) dims in one file.
        url = granules[0][1]
        (lat_blob,) = self._fetch_dap([url], [f"/cell_lat[0:1:{_NROWS - 1}][0]"], auth)
        (lon_blob,) = self._fetch_dap([url], [f"/cell_lon[0][0:1:{_NCOLS - 1}]"], auth)
        lat = _load(lat_blob)["cell_lat"].values.ravel()
        lon = _load(lon_blob)["cell_lon"].values.ravel()
        rows = _index_slice(lat, self.bounds.get("lat_bounds"), "lat_bounds")
        cols = _index_slice(lon, self.bounds.get("lon_bounds"), "lon_bounds")
        if not self.bounds:
            logger.warning("No spatial_subset given; fetching the global SMAP L4 grid.")

        hyperslab = f"[{rows.start}:1:{rows.stop - 1}][{cols.start}:1:{cols.stop - 1}]"
        constraint = ";".join(f"/Geophysical_Data/{v}{hyperslab}" for v in self.variables)
        logins = auth.logins
        urls = [u for _, u in granules]
        blobs: list[bytes] = []
        # Batches of max_workers: aiohttp's timeout also counts time queued for a connection.
        for i in range(0, len(urls), self.max_workers):
            batch = urls[i : i + self.max_workers]
            blobs += self._fetch_dap(batch, [constraint] * len(batch), auth)
        if auth.logins != logins:
            raise APIError(
                "NASA Earthdata session expired while downloading, so granules may be mixed "
                "up; please retry."
            )

        steps = []
        for (time, _), blob in zip(granules, blobs, strict=True):
            ds = _load(blob, group="Geophysical_Data")[self.variables]
            ds = ds.drop_vars(list(ds.coords))
            # All Geophysical_Data variables share the same (y, x) grid.
            dims = dict(zip(ds[self.variables[0]].dims, ("latitude", "longitude"), strict=True))
            steps.append(ds.rename_dims(dims).expand_dims(time=[time.tz_convert(None)]))

        ds = xr.concat(steps, dim="time")
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
            If CMR search fails or Earthdata credentials are missing or rejected.
        ReaperError
            If OPeNDAP requests or processing fail.
        """
        period = "latest" if self.is_latest else f"{self.start_date} to {self.end_date}"
        logger.info(f"Reaping SMAP L4 data for {period}")

        with wrap_errors(ReaperError, "SMAP L4 reaping failed", ReaperError):
            ds = apply_gridded_transformations(self._fetch_data(), self.transformations)
            logger.info(f"Successfully reaped SMAP L4 data: {ds.sizes['time']} time steps")
            return ds
