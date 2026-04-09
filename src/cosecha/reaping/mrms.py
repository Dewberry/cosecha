"""MRMS (Multi-Radar Multi-Sensor) reaper for gridded accumulated precipitation.

This module implements reapers for NOAA MRMS data.
"""

from __future__ import annotations

import gzip
import tempfile
import time as time_mod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

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
        if self.time is None and (self.start_time is None or self.end_time is None):
            raise DateRangeError("Must provide either 'time' or both 'start_time' and 'end_time'.")

        if self.start_time and self.end_time and self.start_time > self.end_time:
            raise DateRangeError("start_time must be <= end_time.")

    def __init__(
        self,
        variable: str = "MultiSensor_QPE_01H_Pass2_00.00",
        time: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        transformations: dict[str, Any] | None = None,
        cache_data: bool = False,
    ) -> None:
        """Initialize MRMSReaper.

        Parameters
        ----------
        variable : str
            MRMS variable name.
        time : str, optional
            A single datetime to fetch, e.g. "2026-01-01 12:00".
        start_time : str, optional
            Start time for the fetch range, e.g. "2026-01-01 00:00".
        end_time : str, optional
            End time for the fetch range, e.g. "2026-01-01 18:00".
        transformations : dict[str, Any], optional
            Optional transformations to apply to the raw data before returning.
        cache_data : bool, optional
            Whether to cache decompressed MRMS files on disk.
        """
        super().__init__()
        self.variable = variable
        try:
            self.time = pd.to_datetime(time) if time is not None else None
            self.start_time = pd.to_datetime(start_time) if start_time is not None else None
            self.end_time = pd.to_datetime(end_time) if end_time is not None else None
        except Exception as e:
            raise DateRangeError(f"Could not parse time parameter: {e}") from e
        self.transformations = transformations
        self.cache_data = cache_data

        self.cache_dir = Path(tempfile.gettempdir()) / "mrms_cache"

        self._validate_params()

        self.aws = s3fs.S3FileSystem(
            anon=True, config_kwargs={"connect_timeout": 30, "read_timeout": 60}
        )

    def _find_available_files(self, times: list[datetime]) -> list[str]:
        files_list = []
        for dt in times:
            yyyymmdd = dt.strftime("%Y%m%d")
            hh = dt.strftime("%H")

            available_files = self.aws.ls(
                f"noaa-mrms-pds/CONUS/{self.variable}/{yyyymmdd}/", refresh=True
            )

            for file in available_files:
                file_hour = file[-15:-13]
                if file_hour == hh:
                    files_list.append(file)

        if not files_list:
            raise APIError(f"No files found for {self.variable} for the requested times.")

        return files_list

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
        times = []
        if self.time is not None:
            times.append(self.time)
        elif self.start_time is not None and self.end_time is not None:
            curr = self.start_time
            while curr <= self.end_time:
                times.append(curr)
                curr += timedelta(hours=1)

        files = self._find_available_files(times)

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
