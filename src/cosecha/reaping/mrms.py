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

import pandas as pd
import s3fs
import xarray as xr

from cosecha._logging import logger
from cosecha._utils import apply_gridded_transformations, to_180, wrap_errors
from cosecha.exceptions import APIError, DateRangeError, ReaperError
from cosecha.reaping.base import GriddedReaper

__all__ = ["MRMSReaper"]


class MRMSReaper(GriddedReaper):
    """Reaper for NOAA MRMS gridded precipitation data."""

    def _validate_params(self) -> None:
        """Validate initialization parameters.

        Raises
        ------
        DateRangeError
            If time parameters are invalid.
        """
        if self.start_date > self.end_date:
            raise DateRangeError("start_date must be <= end_date.")

    def __init__(
        self,
        dates: Literal["latest"] | str | tuple[str, str],  # noqa: PYI051
        variable: str = "MultiSensor_QPE_01H_Pass2_00.00",
        transformations: dict[str, Any] | None = None,
        cache_data: bool = False,
    ) -> None:
        """Initialize MRMSReaper.

        Parameters
        ----------
        dates : Literal["latest"] | str | tuple[str, str]
            "latest" to fetch the most recent available data, a single string for a specific time (e.g., "2026-01-01 00:00Z"),
                or a tuple of (start_time, end_time) to fetch a custom range, e.g. ("2026-01-01 00:00Z", "2026-01-01 18:00Z").
        variable : str
            MRMS variable name.
        transformations : dict[str, Any], optional
            Optional transformations to apply to the raw data before returning.
        cache_data : bool, optional
            Whether to cache decompressed MRMS files on disk.
        """
        super().__init__()
        self.variable = variable
        self.is_latest = dates == "latest"

        try:
            if self.is_latest:
                self.end_date = datetime.now(UTC)
                self.start_date = self.end_date - timedelta(days=1)
            elif isinstance(dates, str):
                self.start_date = pd.to_datetime(dates, utc=True)
                self.end_date = self.start_date
            else:
                self.start_date = pd.to_datetime(dates[0], utc=True)
                self.end_date = pd.to_datetime(dates[1], utc=True)
        except Exception as e:
            raise DateRangeError(f"Could not parse dates parameter: {e}") from e

        self.transformations = transformations
        self.cache_data = cache_data

        self.cache_dir = Path(tempfile.gettempdir()) / "mrms_cache"

        self._validate_params()

        self.aws = s3fs.S3FileSystem(
            anon=True, config_kwargs={"connect_timeout": 30, "read_timeout": 60}
        )

    def _find_available_files(self) -> list[str]:
        files_list = []

        current_date = self.start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = self.end_date.replace(hour=0, minute=0, second=0, microsecond=0)

        while current_date <= end_date:
            yyyymmdd = current_date.strftime("%Y%m%d")

            try:
                available_files = self.aws.ls(
                    f"noaa-mrms-pds/CONUS/{self.variable}/{yyyymmdd}/", refresh=True
                )
            except Exception as e:
                raise APIError(
                    f"Could not list available files for {self.variable} on {yyyymmdd}: {e}"
                ) from e

            for file in available_files:
                with contextlib.suppress(ValueError, IndexError):
                    filename = file.split("/")[-1]
                    # filename looks like: Variable_YYYYMMDD-HHMMSS.grib2.gz
                    timestamp_str = filename.split("_")[-1].split(".")[0]
                    file_dt = datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S").replace(tzinfo=UTC)

                    if self.start_date <= file_dt <= self.end_date:
                        files_list.append(file)

            current_date += timedelta(days=1)

        if not files_list:
            raise APIError(f"No files found for {self.variable} for the requested times.")

        sorted_files = sorted(files_list)
        return [sorted_files[-1]] if self.is_latest else sorted_files

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
        mrms_da = mrms_da.expand_dims("time")
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
