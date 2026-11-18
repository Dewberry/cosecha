"""MRMS (Multi-Radar Multi-Sensor) reaper for gridded accumulated precipitation.

This module implements reapers for NOAA MRMS data.
"""

from __future__ import annotations

import contextlib
import gzip
import tempfile
import time as time_mod
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

import xarray as xr

from cosecha._logging import logger
from cosecha._utils import (
    apply_gridded_transformations,
    import_optional,
    parse_date_range,
    to_180,
    wrap_errors,
)
from cosecha.exceptions import APIError, ReaperError
from cosecha.reaping.base import GriddedReaper

__all__ = ["MRMSReaper"]


def _file_time(path: str) -> datetime:
    """Parse the UTC timestamp from an MRMS file name like ``Variable_YYYYMMDD-HHMMSS.grib2.gz``."""
    timestamp_str = (
        path.rsplit("/", maxsplit=1)[-1].rsplit("_", maxsplit=1)[-1].split(".", maxsplit=1)[0]
    )
    return datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S").replace(tzinfo=UTC)


class MRMSReaper(GriddedReaper):
    """Reaper for NOAA MRMS gridded precipitation data."""

    def __init__(
        self,
        dates: Literal["latest"] | tuple[str, str],
        variable: str = "MultiSensor_QPE_01H_Pass2_00.00",
        transformations: dict[str, Any] | None = None,
        cache_data: bool = False,
        interval: float | None = None,
        tolerance: float = 90,
    ) -> None:
        """Initialize MRMSReaper.

        Parameters
        ----------
        dates : Literal["latest"] | tuple[str, str]
            "latest" to fetch the most recent available data, or a tuple of (start_time, end_time) to fetch a custom range, e.g. ("2026-01-01 00:00Z", "2026-01-01 18:00Z").
                To fetch a single time point, set start_time and end_time to the same value, e.g. ("2026-01-01 00:00Z", "2026-01-01 00:00Z").
        variable : str
            MRMS variable name.
        transformations : dict[str, Any], optional
            Optional transformations to apply to the raw data before returning.
        cache_data : bool, optional
            Whether to cache decompressed MRMS files on disk.
        interval : float, optional
            Minutes between selected files, e.g. 10 to take roughly every 10th minute from
            start_time to end_time. The file nearest each step is used. None (default)
            returns all files in the range. Not supported with dates="latest".
        tolerance : float, optional
            Max seconds between an interval step and the selected file. Default is 90.

        Raises
        ------
        ImportError
            If s3fs, cfgrib or ecCodes is not installed (``pip install 'cosecha[mrms]'``).
        """
        super().__init__()
        import_optional("cfgrib", "mrms")
        s3fs = import_optional("s3fs", "mrms")

        if dates != "latest" and not (isinstance(dates, (tuple, list)) and len(dates) == 2):
            raise ValueError(
                'dates must be "latest" or a sequence of length 2, e.g. ("2026-01-01 00:00Z", "2026-01-01 18:00Z")'
            )

        if interval is not None:
            if dates == "latest":
                raise ValueError('interval is not supported with dates="latest"')
            if interval <= 0:
                raise ValueError("interval must be a positive number of minutes")

        self.variable = variable
        self.is_latest = dates == "latest"
        self.interval = timedelta(minutes=interval) if interval is not None else None
        self.tolerance = timedelta(seconds=tolerance)

        if self.is_latest:
            self.end_date = datetime.now(UTC)
            self.start_date = self.end_date - timedelta(days=1)
        else:
            self.start_date, self.end_date = parse_date_range(dates[0], dates[1])

        self.transformations = transformations
        self.cache_data = cache_data

        self.cache_dir = Path(tempfile.gettempdir()) / "mrms_cache"

        self.aws = s3fs.S3FileSystem(
            anon=True, config_kwargs={"connect_timeout": 30, "read_timeout": 60}
        )

    def _validate_params(self) -> None:
        """Date validation is handled by parse_date_range at construction."""

    def _select_by_interval(
        self, files_list: list[tuple[datetime, str]], interval: timedelta
    ) -> list[str]:
        """Keep the file nearest each interval step, if it's within tolerance."""
        selected: list[str] = []
        skipped: list[str] = []
        target = self.start_date
        while target <= self.end_date:
            file_dt, file = min(files_list, key=lambda x, t=target: abs(x[0] - t))
            if abs(file_dt - target) > self.tolerance:
                skipped.append(target.strftime("%Y-%m-%d %H:%M:%SZ"))
            elif file not in selected:
                selected.append(file)
            target += interval

        if skipped:
            logger.error(
                f"No {self.variable} file within {self.tolerance.total_seconds():g}s of "
                f"{len(skipped)} interval step(s), skipping: {', '.join(skipped)}. "
                "Try to increase tolerance."
            )

        if not selected:
            raise APIError(
                f"No files found for {self.variable} within tolerance of interval. "
                "Try to increase tolerance."
            )
        return selected

    def _find_available_files(self) -> list[str]:
        files_list: list[tuple[datetime, str]] = []

        # With an interval, allow files up to `tolerance` outside the range to match the end steps
        pad = self.tolerance if self.interval else timedelta(0)
        search_start, search_end = self.start_date - pad, self.end_date + pad

        current_date = search_start.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = search_end.replace(hour=0, minute=0, second=0, microsecond=0)

        while current_date <= end_date:
            yyyymmdd = current_date.strftime("%Y%m%d")

            try:
                available_files = self.aws.ls(
                    f"noaa-mrms-pds/CONUS/{self.variable}/{yyyymmdd}/", refresh=True
                )
            except FileNotFoundError:
                logger.debug(f"No directory found for {self.variable} on {yyyymmdd}, skipping.")
                current_date += timedelta(days=1)
                continue
            except Exception as e:
                raise APIError(
                    f"Could not list available files for {self.variable} on {yyyymmdd}: {e}"
                ) from e

            for file in available_files:
                with contextlib.suppress(ValueError, IndexError):
                    file_dt = _file_time(file)
                    if search_start <= file_dt <= search_end:
                        files_list.append((file_dt, file))

            current_date += timedelta(days=1)

        if not files_list:
            raise APIError(f"No files found for {self.variable} for the requested times.")

        files_list.sort()
        if self.is_latest:
            return [files_list[-1][1]]
        if self.interval is None:
            return [file for _, file in files_list]
        return self._select_by_interval(files_list, self.interval)

    def _process_single_file(self, file: str) -> xr.Dataset | None:
        # Deterministic filename for cache based on the S3 file path name
        filename = file.rsplit("/", maxsplit=1)[-1].replace(".gz", "")
        cached_file_path = self.cache_dir / filename

        if self.cache_data and cached_file_path.exists():
            logger.info(f"Loading cached MRMS file from {cached_file_path}")
            data_in = xr.load_dataarray(cached_file_path, engine="cfgrib", decode_timedelta=True)
        else:
            logger.info(f"Fetching MRMS data from S3: {file}")

            max_retries = 3
            compressed_file: bytes | None = None
            for attempt in range(max_retries):
                try:
                    with self.aws.open(file, "rb") as s3_file:
                        compressed_file = cast("bytes", s3_file.read())
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        logger.exception(f"Skipping {file} after {max_retries} failed attempts.")

                    time_mod.sleep(2**attempt)

            if not compressed_file:
                return None

            decompressed_data = gzip.decompress(compressed_file)

            if self.cache_data:
                with cached_file_path.open("wb") as f:
                    f.write(decompressed_data)
                data_in = xr.load_dataarray(
                    cached_file_path, engine="cfgrib", decode_timedelta=True
                )
            else:
                with tempfile.NamedTemporaryFile(suffix=".grib2") as f:
                    f.write(decompressed_data)
                    f.flush()
                    data_in = xr.load_dataarray(f.name, engine="cfgrib", decode_timedelta=True)

        mrms_da = to_180(data_in)
        # GRIB time drops the seconds, so use the exact timestamp from the file name
        mrms_da = mrms_da.expand_dims("time").assign_coords(
            time=[_file_time(file).replace(tzinfo=None)]
        )
        mrms_da = mrms_da.sortby("latitude")

        return mrms_da.to_dataset(name=self.variable)

    def _fetch_data(self) -> xr.Dataset:
        """Fetch MRMS data from S3.

        Returns
        -------
        xr.Dataset
            Raw MRMS gridded data.
        """
        files = self._find_available_files()

        data_arrays = []
        if self.cache_data:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            mrms_ds = self._process_single_file(file)
            if mrms_ds is not None:
                data_arrays.append(mrms_ds)

        if not data_arrays:
            raise APIError("No data could be successfully fetched and processed.")
        elif len(data_arrays) == 1:
            return data_arrays[0]
        return cast(
            "xr.Dataset",
            xr.concat(data_arrays, dim="time", coords="minimal", compat="override"),
        )

    def _reap(self) -> xr.Dataset:
        """Fetch and return MRMS gridded data.

        Returns
        -------
        xr.Dataset
            MRMS gridded dataset.

        Raises
        ------
        APIError
            If data fetching fails.
        """
        logger.info(f"Reaping MRMS data for {self.variable}")

        with wrap_errors(ReaperError, "MRMS reaping failed"):
            ds = self._fetch_data()
            ds = ds.rename({k: k.lower() for k in map(str, ds.dims) if k != k.lower()})

            if self.transformations:
                ds = apply_gridded_transformations(ds, self.transformations)

            logger.info(f"Successfully reaped MRMS data: {len(ds.data_vars)} variables")
            return ds
