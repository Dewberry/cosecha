"""MRMS (Multi-Radar Multi-Sensor) reaper for gridded accumulated precipitation.

This module implements reapers for NOAA MRMS data.
"""

from __future__ import annotations

import gzip
import logging
import os
import tempfile
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import s3fs
import xarray as xr

from cosecha.data_models import HarvestedData
from cosecha.reaping.exceptions import APIError, DateRangeError, ReaperError
from cosecha.reaping.utils import apply_gridded_transformations

if TYPE_CHECKING:
    from cosecha.reaping.base import GriddedReaper

__all__ = ["MRMSReaper"]

logger = logging.getLogger(__name__)


class MRMSReaper:
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
        time: datetime | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        transformations: dict[str, Any] | None = None,
        cache_data: bool = False,
    ) -> None:
        """Initialize MRMSReaper.

        Parameters
        ----------
        variable : str
            MRMS variable name.
        time : datetime, optional
            A single datetime to fetch.
        start_time : datetime, optional
            Start time for the fetch range.
        end_time : datetime, optional
            End time for the fetch range.
        transformations : dict[str, Any], optional
            Optional transformations to apply to the raw data before returning.
        cache_data : bool, optional
            Whether to cache decompressed MRMS files on disk.
        """
        self.variable = variable
        self.time = time
        self.start_time = start_time
        self.end_time = end_time
        self.transformations = transformations
        self.cache_data = cache_data

        # tempdir cache directory
        self.cache_dir = os.path.join(tempfile.gettempdir(), "mrms_cache")

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

    def _to_180(self, ds: xr.DataArray) -> xr.DataArray:
        """Convert longitude from 0-360 to -180 to 180."""
        ds = ds.copy()
        ds["longitude"] = ((ds["longitude"] + 180) % 360) - 180
        return ds.sortby("longitude")

    def fetch_data(self) -> xr.Dataset:
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
            os.makedirs(self.cache_dir, exist_ok=True)

        for file in files:
            # Deterministic filename for cache based on the S3 file path name
            filename = file.split("/")[-1].replace(".gz", ".grib2")
            cached_file_path = os.path.join(self.cache_dir, filename)

            if self.cache_data and os.path.exists(cached_file_path):
                logger.info(f"Loading cached MRMS file from {cached_file_path}")
                data_in = xr.load_dataarray(
                    cached_file_path, engine="cfgrib", decode_timedelta=True
                )
            else:
                logger.info(f"Fetching MRMS data from S3: {file}")

                max_retries = 3
                compressed_file = None
                for attempt in range(max_retries):
                    try:
                        with self.aws.open(file, "rb") as s3_file:
                            compressed_file = s3_file.read()
                        break
                    except Exception as e:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}")
                        if attempt == max_retries - 1:
                            logger.exception(
                                f"Skipping {file} after {max_retries} failed attempts."
                            )
                        import time as time_mod

                        time_mod.sleep(2**attempt)

                if not compressed_file:
                    continue

                decompressed_data = gzip.decompress(compressed_file)

                if self.cache_data:
                    with open(cached_file_path, "wb") as f:
                        f.write(decompressed_data)
                    data_in = xr.load_dataarray(
                        cached_file_path, engine="cfgrib", decode_timedelta=True
                    )
                else:
                    with tempfile.NamedTemporaryFile(suffix=".grib2") as f:
                        f.write(decompressed_data)
                        f.flush()
                        data_in = xr.load_dataarray(f.name, engine="cfgrib", decode_timedelta=True)

            mrms_da = self._to_180(data_in)

            mrms_da = mrms_da.expand_dims("time")
            mrms_da = mrms_da.sortby("latitude")

            mrms_ds = mrms_da.to_dataset(name=self.variable)
            if self.transformations:
                mrms_ds = apply_gridded_transformations(mrms_ds, self.transformations)
            data_arrays.append(mrms_ds)

        if not data_arrays:
            raise APIError("No data could be successfully fetched and processed.")
        elif len(data_arrays) == 1:
            return data_arrays[0]
        else:
            return xr.concat(data_arrays, dim="time")

    def reap(self) -> HarvestedData:
        """Fetch and return MRMS gridded data.

        Returns
        -------
        HarvestedData
            Container with xarray Dataset, source name, timestamp, variables,
            and metadata.

        Raises
        ------
        APIError
            If data fetching fails.
        """
        logger.info(f"Reaping MRMS data for {self.variable}")

        try:
            ds = self.fetch_data()

            # Ensure all dimension names are lowercase
            ds = ds.rename({k: k.lower() for k in ds.dims if k != k.lower()})

            variable_names = list(ds.data_vars)

            metadata = {
                "source_name": "MRMS",
                "timestamp": datetime.now(UTC),
                "variable_names": variable_names,
                "variable": self.variable,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
            }

            harvested = HarvestedData(
                data=ds,
                source_name="MRMS",
                timestamp=metadata["timestamp"],
                variable_names=variable_names,
                metadata=metadata,
            )

            logger.info(f"Successfully reaped MRMS data: {len(variable_names)} variables")
            return harvested

        except Exception as e:
            logger.exception(f"Failed to reap MRMS data: {e}")
            raise ReaperError(f"MRMS reaping failed: {e}") from e


_: type[GriddedReaper] = MRMSReaper
