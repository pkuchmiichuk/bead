"""Tests for behavioral analytics models."""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

import pandas as pd
import polars as pl
import pytest
from slopit.schemas import FocusMetrics, KeystrokeMetrics, TimingMetrics

from bead.behavioral.analytics import (
    AnalysisFlag,
    AnalyticsCollection,
    JudgmentAnalytics,
    ParticipantBehavioralSummary,
)


@pytest.fixture
def sample_keystroke_metrics() -> dict[str, float | int]:
    """Create sample keystroke metrics dict (slopit -> JSON shape)."""
    return KeystrokeMetrics(
        total_keystrokes=50,
        printable_keystrokes=45,
        deletions=5,
        mean_iki=150.0,
        std_iki=30.0,
        median_iki=140.0,
        pause_count=2,
        product_process_ratio=0.9,
    ).model_dump()


@pytest.fixture
def sample_focus_metrics() -> dict[str, float | int]:
    """Create sample focus metrics dict."""
    return FocusMetrics(
        blur_count=1,
        total_blur_duration=500.0,
        hidden_count=0,
        total_hidden_duration=0.0,
    ).model_dump()


@pytest.fixture
def sample_timing_metrics() -> dict[str, float | int]:
    """Create sample timing metrics dict."""
    return TimingMetrics(
        first_keystroke_latency=1200.0,
        total_response_time=5000.0,
        characters_per_minute=120.0,
    ).model_dump()


@pytest.fixture
def sample_flag() -> AnalysisFlag:
    """Create sample analysis flag."""
    return AnalysisFlag(
        type="rapid_response",
        severity="medium",
        message="Response time below threshold",
        metadata={"analyzer": "timing", "confidence": 0.85},
    )


@pytest.fixture
def sample_analytics(
    sample_keystroke_metrics: dict,
    sample_focus_metrics: dict,
    sample_timing_metrics: dict,
) -> JudgmentAnalytics:
    """Create sample judgment analytics."""
    return JudgmentAnalytics(
        item_id=uuid4(),
        participant_id="participant_001",
        trial_index=0,
        session_id="session_001",
        response_value=5,
        response_time_ms=2500,
        keystroke_metrics=sample_keystroke_metrics,
        focus_metrics=sample_focus_metrics,
        timing_metrics=sample_timing_metrics,
        paste_event_count=0,
    )


class TestJudgmentAnalytics:
    """Tests for JudgmentAnalytics model."""

    def test_creation_minimal(self) -> None:
        """Test creating analytics with minimal fields."""
        analytics = JudgmentAnalytics(
            item_id=uuid4(),
            participant_id="p001",
            trial_index=0,
            session_id="s001",
            response_time_ms=1000,
        )
        assert analytics.is_flagged is False
        assert analytics.has_paste_events is False
        assert analytics.keystroke_metrics is None

    def test_creation_with_metrics(
        self,
        sample_keystroke_metrics: dict,
    ) -> None:
        """Test creating analytics with behavioral metrics."""
        analytics = JudgmentAnalytics(
            item_id=uuid4(),
            participant_id="p001",
            trial_index=0,
            session_id="s001",
            response_time_ms=2000,
            keystroke_metrics=sample_keystroke_metrics,
        )
        assert analytics.keystroke_metrics is not None
        assert analytics.keystroke_metrics["total_keystrokes"] == 50

    def test_has_paste_events(self) -> None:
        """Test paste event detection."""
        analytics = JudgmentAnalytics(
            item_id=uuid4(),
            participant_id="p001",
            trial_index=0,
            session_id="s001",
            response_time_ms=1000,
            paste_event_count=2,
        )
        assert analytics.has_paste_events is True

    def test_is_flagged_with_flags(self, sample_flag: AnalysisFlag) -> None:
        """Test flag detection."""
        analytics = JudgmentAnalytics(
            item_id=uuid4(),
            participant_id="p001",
            trial_index=0,
            session_id="s001",
            response_time_ms=500,
            flags=[sample_flag],
            max_severity="medium",
        )
        assert analytics.is_flagged is True
        assert analytics.get_flag_types() == ("rapid_response",)

    def test_get_flag_types_empty(self) -> None:
        """Test flag types with no flags."""
        analytics = JudgmentAnalytics(
            item_id=uuid4(),
            participant_id="p001",
            trial_index=0,
            session_id="s001",
            response_time_ms=1000,
        )
        assert analytics.get_flag_types() == ()


class TestParticipantBehavioralSummary:
    """Tests for ParticipantBehavioralSummary model."""

    def test_creation(self) -> None:
        """Test creating a participant summary."""
        summary = ParticipantBehavioralSummary(
            participant_id="p001",
            session_id="s001",
            total_judgments=50,
            flagged_judgments=5,
            mean_response_time_ms=2500.0,
        )
        assert summary.flag_rate == 0.1

    def test_flag_rate_zero_judgments(self) -> None:
        """Test flag rate with zero judgments."""
        summary = ParticipantBehavioralSummary(
            participant_id="p001",
            session_id="s001",
            total_judgments=0,
            flagged_judgments=0,
            mean_response_time_ms=0.0,
        )
        assert summary.flag_rate == 0.0

    def test_has_paste_events(self) -> None:
        """Test paste event detection in summary."""
        summary = ParticipantBehavioralSummary(
            participant_id="p001",
            session_id="s001",
            total_judgments=10,
            mean_response_time_ms=2000.0,
            total_paste_events=3,
        )
        assert summary.has_paste_events is True


class TestAnalyticsCollection:
    """Tests for AnalyticsCollection."""

    def test_creation_empty(self) -> None:
        """Test creating empty collection."""
        collection = AnalyticsCollection(name="test")
        assert len(collection) == 0

    def test_add_analytics(self, sample_analytics: JudgmentAnalytics) -> None:
        """Test adding analytics to collection."""
        collection = AnalyticsCollection(name="test")
        collection = collection.with_analytics(sample_analytics)
        assert len(collection) == 1

    def test_add_many(self) -> None:
        """Test adding multiple analytics."""
        collection = AnalyticsCollection(name="test")
        analytics_list = [
            JudgmentAnalytics(
                item_id=uuid4(),
                participant_id=f"p{i}",
                trial_index=i,
                session_id="s001",
                response_time_ms=1000 + i * 100,
            )
            for i in range(5)
        ]
        collection = collection.with_many(analytics_list)
        assert len(collection) == 5

    def test_get_by_participant(self) -> None:
        """Test filtering by participant."""
        collection = AnalyticsCollection(name="test")
        for i in range(3):
            collection = collection.with_analytics(
                JudgmentAnalytics(
                    item_id=uuid4(),
                    participant_id="p001",
                    trial_index=i,
                    session_id="s001",
                    response_time_ms=1000,
                )
            )
        collection = collection.with_analytics(
            JudgmentAnalytics(
                item_id=uuid4(),
                participant_id="p002",
                trial_index=0,
                session_id="s002",
                response_time_ms=1000,
            )
        )

        p001_records = collection.get_by_participant("p001")
        assert len(p001_records) == 3

    def test_get_by_item(self) -> None:
        """Test filtering by item."""
        item_id = uuid4()
        collection = AnalyticsCollection(name="test")
        collection = collection.with_analytics(
            JudgmentAnalytics(
                item_id=item_id,
                participant_id="p001",
                trial_index=0,
                session_id="s001",
                response_time_ms=1000,
            )
        )
        collection = collection.with_analytics(
            JudgmentAnalytics(
                item_id=item_id,
                participant_id="p002",
                trial_index=0,
                session_id="s002",
                response_time_ms=1200,
            )
        )
        collection = collection.with_analytics(
            JudgmentAnalytics(
                item_id=uuid4(),  # Different item
                participant_id="p001",
                trial_index=1,
                session_id="s001",
                response_time_ms=1100,
            )
        )

        item_records = collection.get_by_item(item_id)
        assert len(item_records) == 2

    def test_get_participant_ids(self) -> None:
        """Test getting unique participant IDs."""
        collection = AnalyticsCollection(name="test")
        for pid in ["p001", "p002", "p001", "p003"]:
            collection = collection.with_analytics(
                JudgmentAnalytics(
                    item_id=uuid4(),
                    participant_id=pid,
                    trial_index=0,
                    session_id="s001",
                    response_time_ms=1000,
                )
            )

        ids = collection.get_participant_ids()
        assert set(ids) == {"p001", "p002", "p003"}

    def test_filter_flagged_include(self, sample_flag: AnalysisFlag) -> None:
        """Test filtering to include only flagged records."""
        collection = AnalyticsCollection(name="test")

        # Add unflagged
        collection = collection.with_analytics(
            JudgmentAnalytics(
                item_id=uuid4(),
                participant_id="p001",
                trial_index=0,
                session_id="s001",
                response_time_ms=2000,
            )
        )

        # Add flagged
        collection = collection.with_analytics(
            JudgmentAnalytics(
                item_id=uuid4(),
                participant_id="p002",
                trial_index=0,
                session_id="s002",
                response_time_ms=500,
                flags=[sample_flag],
                max_severity="medium",
            )
        )

        flagged = collection.filter_flagged(exclude_flagged=False)
        assert len(flagged) == 1
        assert flagged.analytics[0].participant_id == "p002"

    def test_filter_flagged_exclude(self, sample_flag: AnalysisFlag) -> None:
        """Test filtering to exclude flagged records."""
        collection = AnalyticsCollection(name="test")

        # Add unflagged
        collection = collection.with_analytics(
            JudgmentAnalytics(
                item_id=uuid4(),
                participant_id="p001",
                trial_index=0,
                session_id="s001",
                response_time_ms=2000,
            )
        )

        # Add flagged
        collection = collection.with_analytics(
            JudgmentAnalytics(
                item_id=uuid4(),
                participant_id="p002",
                trial_index=0,
                session_id="s002",
                response_time_ms=500,
                flags=[sample_flag],
                max_severity="medium",
            )
        )

        unflagged = collection.filter_flagged(exclude_flagged=True)
        assert len(unflagged) == 1
        assert unflagged.analytics[0].participant_id == "p001"

    def test_filter_flagged_by_severity(self) -> None:
        """Test filtering by minimum severity."""
        collection = AnalyticsCollection(name="test")

        # Low severity flag
        low_flag = AnalysisFlag(
            type="minor_issue",
            severity="low",
            message="Minor focus issue",
            metadata={"analyzer": "focus"},
        )

        # High severity flag
        high_flag = AnalysisFlag(
            type="major_issue",
            severity="high",
            message="Large paste detected",
            metadata={"analyzer": "paste"},
        )

        collection = collection.with_analytics(
            JudgmentAnalytics(
                item_id=uuid4(),
                participant_id="p001",
                trial_index=0,
                session_id="s001",
                response_time_ms=1000,
                flags=[low_flag],
                max_severity="low",
            )
        )

        collection = collection.with_analytics(
            JudgmentAnalytics(
                item_id=uuid4(),
                participant_id="p002",
                trial_index=0,
                session_id="s002",
                response_time_ms=1000,
                flags=[high_flag],
                max_severity="high",
            )
        )

        # Filter for high severity only
        high_only = collection.filter_flagged(
            min_severity="high", exclude_flagged=False
        )
        assert len(high_only) == 1
        assert high_only.analytics[0].participant_id == "p002"

    def test_get_participant_summaries(
        self,
        sample_keystroke_metrics: dict,
        sample_focus_metrics: dict,
        sample_flag: AnalysisFlag,
    ) -> None:
        """Test generating participant summaries."""
        collection = AnalyticsCollection(name="test")

        # Add multiple records for one participant
        for i in range(3):
            collection = collection.with_analytics(
                JudgmentAnalytics(
                    item_id=uuid4(),
                    participant_id="p001",
                    trial_index=i,
                    session_id="s001",
                    response_time_ms=2000 + i * 100,
                    keystroke_metrics=sample_keystroke_metrics,
                    focus_metrics=sample_focus_metrics,
                    flags=[sample_flag] if i == 0 else [],
                    max_severity="medium" if i == 0 else None,
                )
            )

        summaries = collection.get_participant_summaries()
        assert len(summaries) == 1

        summary = summaries[0]
        assert summary.participant_id == "p001"
        assert summary.total_judgments == 3
        assert summary.flagged_judgments == 1
        assert summary.total_keystrokes == 150  # 50 * 3
        assert summary.flag_rate == pytest.approx(1 / 3)

    def test_jsonl_roundtrip(self, sample_analytics: JudgmentAnalytics) -> None:
        """Test saving and loading from JSONL."""
        collection = AnalyticsCollection(name="test")
        collection = collection.with_analytics(sample_analytics)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "analytics.jsonl"
            collection.to_jsonl(path)

            loaded = AnalyticsCollection.from_jsonl(path, name="loaded")
            assert len(loaded) == 1
            assert loaded.analytics[0].participant_id == "participant_001"

    def test_to_dataframe_pandas(self, sample_analytics: JudgmentAnalytics) -> None:
        """Test converting to pandas DataFrame."""
        collection = AnalyticsCollection(name="test")
        collection = collection.with_analytics(sample_analytics)
        df = collection.to_dataframe(backend="pandas")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "item_id" in df.columns
        assert "keystroke_mean_iki" in df.columns

    def test_to_dataframe_polars(self, sample_analytics: JudgmentAnalytics) -> None:
        """Test converting to polars DataFrame."""
        collection = AnalyticsCollection(name="test")
        collection = collection.with_analytics(sample_analytics)
        df = collection.to_dataframe(backend="polars")
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 1
        assert "item_id" in df.columns

    def test_to_dataframe_empty(self) -> None:
        """Test converting empty collection to DataFrame."""
        collection = AnalyticsCollection(name="test")

        df_pandas = collection.to_dataframe(backend="pandas")
        assert len(df_pandas) == 0

        df_polars = collection.to_dataframe(backend="polars")
        assert len(df_polars) == 0

    def test_to_dataframe_without_metrics(
        self, sample_analytics: JudgmentAnalytics
    ) -> None:
        """Test DataFrame without metrics columns."""
        collection = AnalyticsCollection(name="test")
        collection = collection.with_analytics(sample_analytics)
        df = collection.to_dataframe(include_metrics=False)
        assert "keystroke_mean_iki" not in df.columns

    def test_to_dataframe_without_flags(
        self, sample_analytics: JudgmentAnalytics
    ) -> None:
        """Test DataFrame without flag columns."""
        collection = AnalyticsCollection(name="test")
        collection = collection.with_analytics(sample_analytics)
        df = collection.to_dataframe(include_flags=False)
        assert "is_flagged" not in df.columns
