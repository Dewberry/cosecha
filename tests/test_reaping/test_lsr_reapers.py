"""Tests for LSR reaper."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pytest

from cosecha.exceptions import APIError, DataNotFoundError, DateRangeError
from cosecha.reaping import lsr as lsr_mod
from cosecha.reaping.lsr import LSRReaper


def _feature(
    valid: str,
    typetext: str,
    *,
    type_code: str = "F",
    magf: float | None = None,
    magnitude: str = "",
    unit: str | None = None,
    wfo: str = "BOU",
    county: str = "Denver",
    state: str = "CO",
    st: str = "CO",
    city: str = "Denver",
    source: str = "Public",
    remark: str | None = None,
    qualifier: str | None = None,
    product_id: str = "202607011200-KBOU-NWUS55-LSRBOU",
    lon: float = -105.0,
    lat: float = 39.7,
) -> dict:
    """Build a realistic IEM LSR GeoJSON feature."""
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {
            "valid": valid,
            "type": type_code,
            "typetext": typetext,
            "magf": magf,
            "magnitude": magnitude,
            "unit": unit,
            "wfo": wfo,
            "county": county,
            "state": state,
            "st": st,
            "city": city,
            "source": source,
            "remark": remark,
            "qualifier": qualifier,
            "product_id": product_id,
            "lon": lon,
            "lat": lat,
        },
    }


SAMPLE_FEATURES = [
    _feature(
        "2026-07-01T12:00:00Z",
        "FLASH FLOOD",
        remark="Water over road",
        source="Emergency Manager",
        product_id="202607011230-KBOU-NWUS55-LSRBOU",
    ),
    _feature(
        "2026-07-01T14:00:00Z",
        "HEAVY RAIN",
        type_code="R",
        magf=2.5,
        magnitude="2.5",
        unit="INCH",
        county="Arapahoe",
        city="Aurora",
        source="CoCoRaHS",
        remark="Heavy rainfall",
        qualifier="M",
        product_id="202607011430-KBOU-NWUS55-LSRBOU",
        lon=-104.5,
        lat=39.5,
    ),
    _feature(
        "2026-07-01T15:00:00Z",
        "HAIL",
        type_code="H",
        magf=1.0,
        magnitude="1",
        unit="INCH",
        county="Boulder",
        city="Boulder",
        remark="Golf ball hail",
        qualifier="E",
        product_id="202607011530-KBOU-NWUS55-LSRBOU",
        lon=-106.0,
        lat=40.0,
    ),
    _feature(
        "2026-07-01T16:00:00Z",
        "FLASH FLOOD",
        wfo="GLD",
        county="Baca",
        state="KS",
        st="KS",
        city="Springfield",
        source="Law Enforcement",
        remark="Water over road",
        product_id="202607011630-KGLD-NWUS55-LSRGLD",
        lon=-102.0,
        lat=37.5,
    ),
]

SAMPLE_GEOJSON = {"type": "FeatureCollection", "features": SAMPLE_FEATURES}


class TestLSRReaper:
    """Tests for LSRReaper."""

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def test_initialization_valid(self):
        """Test valid initialization with all parameters."""
        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
            wfos=["BOU", "GJT"],
            event_types=["FLASH FLOOD", "HEAVY RAIN"],
            state="CO",
        )
        assert reaper.start_date == pd.Timestamp("2026-07-01T00:00:00", tz="UTC")
        assert reaper.end_date == pd.Timestamp("2026-07-02T00:00:00", tz="UTC")
        assert reaper.wfos == ["BOU", "GJT"]
        assert reaper.event_types == ["FLASH FLOOD", "HEAVY RAIN"]
        assert reaper.state == "CO"

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        reaper = LSRReaper(
            start_date="2026-07-01",
            end_date="2026-07-02",
        )
        assert reaper.wfos is None
        assert reaper.state is None
        assert reaper.event_types is None

    def test_invalid_date_range(self):
        """Test initialization fails with start > end."""
        with pytest.raises(DateRangeError):
            LSRReaper(
                start_date="2026-07-02",
                end_date="2026-07-01",
            )

    def test_invalid_date_format(self):
        """Test initialization fails with unparsable date."""
        with pytest.raises(DateRangeError, match="Could not parse date"):
            LSRReaper(
                start_date="not-a-date",
                end_date="2026-07-02",
            )

    def test_offset_input_converted_to_utc(self):
        """Test that offset-aware input is stored as UTC."""
        reaper = LSRReaper(
            start_date="2026-07-01T13:00:00-05:00",
            end_date="2026-07-01T15:00:00-05:00",
        )
        assert reaper.start_date == pd.Timestamp("2026-07-01T18:00:00", tz="UTC")
        assert reaper.end_date == pd.Timestamp("2026-07-01T20:00:00", tz="UTC")

    # ------------------------------------------------------------------
    # URL construction
    # ------------------------------------------------------------------

    def test_build_url_with_wfos(self):
        """Test URL construction includes WFOs."""
        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
            wfos=["BOU", "GJT"],
        )
        url = reaper._build_url()
        assert "sts=202607010000" in url
        assert "ets=202607020000" in url
        assert "wfos=BOU%2CGJT" in url or "wfos=BOU,GJT" in url

    def test_build_url_without_wfos(self):
        """Test URL construction without WFOs."""
        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
        )
        url = reaper._build_url()
        assert "wfos" not in url

    def test_build_url_sends_states_not_state(self):
        """Test URL sends 'states' (plural) for server-side filter."""
        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
            state="CO",
        )
        url = reaper._build_url()
        assert "states=CO" in url
        # Must NOT send singular "state=" (IEM silently ignores it)
        assert "state=" not in url.replace("states=", "")

    def test_build_url_offset_input_formats_utc(self):
        """Test that offset-aware input is formatted as UTC in the URL."""
        reaper = LSRReaper(
            start_date="2026-07-01T13:00:00-05:00",
            end_date="2026-07-01T15:00:00-05:00",
        )
        url = reaper._build_url()
        assert "sts=202607011800" in url
        assert "ets=202607012000" in url

    # ------------------------------------------------------------------
    # Fetch / reap (mocked)
    # ------------------------------------------------------------------

    @patch("cosecha.reaping.lsr.tiny_retriever.fetch")
    def test_reap_success(self, mock_fetch):
        """Test successful data retrieval with filters."""
        mock_fetch.return_value = [SAMPLE_GEOJSON]

        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
            wfos=["BOU"],
            event_types=["FLASH FLOOD", "HEAVY RAIN"],
            state="CO",
        )
        df = reaper.reap()

        assert isinstance(df, pd.DataFrame)
        # HAIL and KS features should be filtered out
        assert len(df) == 2
        assert set(df["event_type"]) == {"FLASH FLOOD", "HEAVY RAIN"}
        assert all(df["state"] == "CO")

    @patch("cosecha.reaping.lsr.tiny_retriever.fetch")
    def test_reap_no_state_filter(self, mock_fetch):
        """Test retrieval without state filter returns more results."""
        mock_fetch.return_value = [SAMPLE_GEOJSON]

        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
            event_types=["FLASH FLOOD", "HEAVY RAIN"],
        )
        df = reaper.reap()

        assert len(df) == 3  # Both CO and KS flash floods + heavy rain

    @patch("cosecha.reaping.lsr.tiny_retriever.fetch")
    def test_reap_all_event_types(self, mock_fetch):
        """Test retrieval with no event_types filter returns all types."""
        mock_fetch.return_value = [SAMPLE_GEOJSON]

        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
        )
        df = reaper.reap()

        assert len(df) == 4  # All features including HAIL
        assert set(df["event_type"]) == {"FLASH FLOOD", "HEAVY RAIN", "HAIL"}

    @patch("cosecha.reaping.lsr.tiny_retriever.fetch")
    def test_reap_empty_features(self, mock_fetch):
        """Test raises DataNotFoundError when no features returned."""
        mock_fetch.return_value = [{"type": "FeatureCollection", "features": []}]

        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
        )
        with pytest.raises(DataNotFoundError, match="no features"):
            reaper.reap()

    @patch("cosecha.reaping.lsr.tiny_retriever.fetch")
    def test_reap_no_matching_features(self, mock_fetch):
        """Test raises DataNotFoundError when no features match filters."""
        mock_fetch.return_value = [SAMPLE_GEOJSON]

        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
            event_types=["TORNADO"],
        )
        with pytest.raises(DataNotFoundError, match="No LSR features matched"):
            reaper.reap()

    @patch("cosecha.reaping.lsr.tiny_retriever.fetch")
    def test_reap_api_error(self, mock_fetch):
        """Test wraps fetch errors as APIError."""
        mock_fetch.side_effect = Exception("Connection timeout")

        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
        )
        with pytest.raises(APIError, match="Failed to fetch LSR data"):
            reaper.reap()

    @patch("cosecha.reaping.lsr.tiny_retriever.fetch")
    def test_reap_with_transformations(self, mock_fetch):
        """Test transformations are applied to result."""
        mock_fetch.return_value = [SAMPLE_GEOJSON]

        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
            event_types=["FLASH FLOOD", "HEAVY RAIN"],
            state="CO",
            transformations={"rename_columns": {"event_type": "type"}},
        )
        df = reaper.reap()
        assert "type" in df.columns
        assert "event_type" not in df.columns

    @patch("cosecha.reaping.lsr.tiny_retriever.fetch")
    def test_coordinates_from_properties(self, mock_fetch):
        """Test lat/lon are sourced from properties, not geometry."""
        mock_fetch.return_value = [SAMPLE_GEOJSON]

        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
            event_types=["FLASH FLOOD"],
            state="CO",
        )
        df = reaper.reap()
        assert df.iloc[0]["longitude"] == -105.0
        assert df.iloc[0]["latitude"] == 39.7

    # ------------------------------------------------------------------
    # Output dtypes
    # ------------------------------------------------------------------

    @patch("cosecha.reaping.lsr.tiny_retriever.fetch")
    def test_magnitude_is_float(self, mock_fetch):
        """Test magnitude column is float64 (sourced from magf, not string magnitude)."""
        mock_fetch.return_value = [SAMPLE_GEOJSON]

        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
        )
        df = reaper.reap()
        assert df["magnitude"].dtype == "float64"
        assert df.loc[df["event_type"] == "HEAVY RAIN", "magnitude"].iloc[0] == 2.5

    @patch("cosecha.reaping.lsr.tiny_retriever.fetch")
    def test_valid_is_utc_datetime(self, mock_fetch):
        """Test valid column is tz-aware UTC datetime."""
        mock_fetch.return_value = [SAMPLE_GEOJSON]

        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
        )
        df = reaper.reap()
        assert df["valid"].dt.tz is not None
        assert str(df["valid"].dt.tz) == "UTC"

    @patch("cosecha.reaping.lsr.tiny_retriever.fetch")
    def test_product_id_present(self, mock_fetch):
        """Test product_id column is included in output."""
        mock_fetch.return_value = [SAMPLE_GEOJSON]

        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
        )
        df = reaper.reap()
        assert "product_id" in df.columns
        assert df["product_id"].notna().all()

    @patch("cosecha.reaping.lsr.tiny_retriever.fetch")
    def test_all_null_magnitude_batch_has_arrow_float(self, mock_fetch):
        """Test that filtering to events with no magnitude keeps float64, not Arrow null."""
        mock_fetch.return_value = [SAMPLE_GEOJSON]

        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
            event_types=["FLASH FLOOD"],
        )
        df = reaper.reap()
        # All flash-flood magnitudes are None/NaN, but dtype must stay float
        assert df["magnitude"].isna().all()
        assert df["magnitude"].dtype == "float64"
        # Must not raise ArrowTypeError
        table = pa.Table.from_pandas(df)
        mag_type = table.schema.field("magnitude").type
        assert mag_type != pa.null()

    # ------------------------------------------------------------------
    # Row-cap bisection
    # ------------------------------------------------------------------

    @patch("cosecha.reaping.lsr.tiny_retriever.fetch")
    def test_fetch_windows_bisects_on_cap(self, mock_fetch):
        """Test that a response hitting the row cap triggers bisection."""
        capped_features = [_feature(f"2026-07-01T{h:02d}:00:00Z", "HAIL") for h in range(10)]
        sub_features_a = capped_features[:5]
        sub_features_b = capped_features[5:]

        # First call hits cap, second call returns two sub-cap responses
        mock_fetch.side_effect = [
            [{"type": "FeatureCollection", "features": capped_features}],
            [
                {"type": "FeatureCollection", "features": sub_features_a},
                {"type": "FeatureCollection", "features": sub_features_b},
            ],
        ]

        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
        )

        # Use a lower cap to trigger bisection with our small fixture
        original_cap = lsr_mod._IEM_ROW_CAP
        lsr_mod._IEM_ROW_CAP = len(capped_features)
        try:
            features = reaper._fetch_windows([(reaper.start_date, reaper.end_date)])
        finally:
            lsr_mod._IEM_ROW_CAP = original_cap

        assert len(features) == 10
        assert mock_fetch.call_count == 2
        # Second call should have 2 URLs (the bisected halves)
        assert len(mock_fetch.call_args_list[1][0][0]) == 2

    @patch("cosecha.reaping.lsr.tiny_retriever.fetch")
    def test_fetch_windows_unsplittable_raises(self, mock_fetch):
        """Test APIError when a 1-minute window hits the cap."""
        capped_features = [_feature("2026-07-01T12:00:00Z", "HAIL")] * 5

        mock_fetch.return_value = [{"type": "FeatureCollection", "features": capped_features}]

        reaper = LSRReaper(
            start_date="2026-07-01T12:00:00Z",
            end_date="2026-07-01T12:01:00Z",
        )

        original_cap = lsr_mod._IEM_ROW_CAP
        lsr_mod._IEM_ROW_CAP = len(capped_features)
        try:
            with pytest.raises(APIError, match="cannot be split further"):
                reaper._fetch_windows([(reaper.start_date, reaper.end_date)])
        finally:
            lsr_mod._IEM_ROW_CAP = original_cap

    @patch("cosecha.reaping.lsr.tiny_retriever.fetch")
    def test_fetch_windows_sub_cap_no_bisection(self, mock_fetch):
        """Test that a response under the cap is returned without bisection."""
        mock_fetch.return_value = [SAMPLE_GEOJSON]

        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
        )
        features = reaper._fetch_windows([(reaper.start_date, reaper.end_date)])
        assert len(features) == 4
        assert mock_fetch.call_count == 1
