"""Tests for behavioral analytics merging utilities."""

from __future__ import annotations

# Use fixed UUIDs for consistent testing
from uuid import UUID, uuid4

import pandas as pd
import polars as pl
import pytest

from bead.behavioral.analytics import (
    AnalysisFlag,
    AnalyticsCollection,
    JudgmentAnalytics,
)
from bead.behavioral.merging import (
    filter_flagged_judgments,
    get_exclusion_list,
    merge_behavioral_analytics,
)

ITEM_IDS = [
    UUID("11111111-1111-1111-1111-111111111111"),
    UUID("22222222-2222-2222-2222-222222222222"),
    UUID("33333333-3333-3333-3333-333333333333"),
]


@pytest.fixture
def sample_judgments_df() -> pd.DataFrame:
    """Create sample judgments DataFrame.

    Polars cannot infer Arrow types for UUID values, so the dataframe
    uses string item_ids.
    """
    return pd.DataFrame(
        {
            "item_id": [str(i) for i in ITEM_IDS],
            "participant_id": ["p001", "p001", "p002"],
            "response": [5, 3, 4],
        }
    )


@pytest.fixture
def sample_analytics_collection() -> AnalyticsCollection:
    """Create sample analytics collection matching judgments."""
    collection = AnalyticsCollection(name="test")

    # Add analytics for each judgment
    collection = collection.with_analytics(
        JudgmentAnalytics(
            item_id=ITEM_IDS[0],
            participant_id="p001",
            trial_index=0,
            session_id="s001",
            response_time_ms=2000,
        )
    )
    collection = collection.with_analytics(
        JudgmentAnalytics(
            item_id=ITEM_IDS[1],
            participant_id="p001",
            trial_index=1,
            session_id="s001",
            response_time_ms=2500,
        )
    )
    collection = collection.with_analytics(
        JudgmentAnalytics(
            item_id=ITEM_IDS[2],
            participant_id="p002",
            trial_index=0,
            session_id="s002",
            response_time_ms=1800,
        )
    )

    return collection


class TestMergeBehavioralAnalytics:
    """Tests for merge_behavioral_analytics function."""

    def test_merge_pandas(
        self,
        sample_judgments_df: pd.DataFrame,
        sample_analytics_collection: AnalyticsCollection,
    ) -> None:
        """Test merging with pandas DataFrame."""
        merged = merge_behavioral_analytics(
            sample_judgments_df,
            sample_analytics_collection,
        )

        assert isinstance(merged, pd.DataFrame)
        assert len(merged) == 3
        assert "response_time_ms" in merged.columns
        assert "response" in merged.columns

    def test_merge_polars(
        self,
        sample_judgments_df: pd.DataFrame,
        sample_analytics_collection: AnalyticsCollection,
    ) -> None:
        """Test merging with polars DataFrame."""
        judgments_pl = pl.from_pandas(sample_judgments_df)

        merged = merge_behavioral_analytics(
            judgments_pl,
            sample_analytics_collection,
        )

        assert isinstance(merged, pl.DataFrame)
        assert len(merged) == 3
        assert "response_time_ms" in merged.columns

    def test_merge_with_metrics(
        self,
        sample_judgments_df: pd.DataFrame,
        sample_analytics_collection: AnalyticsCollection,
    ) -> None:
        """Test that metrics columns are included."""
        merged = merge_behavioral_analytics(
            sample_judgments_df,
            sample_analytics_collection,
            include_metrics=True,
        )

        # Check for metrics columns (they'll be None but should exist)
        assert "keystroke_total" in merged.columns
        assert "focus_blur_count" in merged.columns

    def test_merge_without_metrics(
        self,
        sample_judgments_df: pd.DataFrame,
        sample_analytics_collection: AnalyticsCollection,
    ) -> None:
        """Test merging without metrics columns."""
        merged = merge_behavioral_analytics(
            sample_judgments_df,
            sample_analytics_collection,
            include_metrics=False,
        )

        assert "keystroke_total" not in merged.columns

    def test_merge_with_flags(
        self,
        sample_judgments_df: pd.DataFrame,
        sample_analytics_collection: AnalyticsCollection,
    ) -> None:
        """Test that flag columns are included."""
        merged = merge_behavioral_analytics(
            sample_judgments_df,
            sample_analytics_collection,
            include_flags=True,
        )

        assert "is_flagged" in merged.columns
        assert "flag_count" in merged.columns


class TestFilterFlaggedJudgments:
    """Tests for filter_flagged_judgments function."""

    @pytest.fixture
    def collection_with_flags(self) -> AnalyticsCollection:
        """Create collection with some flagged records."""
        collection = AnalyticsCollection(name="test")

        # Unflagged record
        collection = collection.with_analytics(
            JudgmentAnalytics(
                item_id=ITEM_IDS[0],
                participant_id="p001",
                trial_index=0,
                session_id="s001",
                response_time_ms=2000,
            )
        )

        # Flagged record
        flag = AnalysisFlag(
            type="rapid_response",
            severity="medium",
            message="Too fast",
        )
        collection = collection.with_analytics(
            JudgmentAnalytics(
                item_id=ITEM_IDS[1],
                participant_id="p001",
                trial_index=1,
                session_id="s001",
                response_time_ms=500,
                flags=[flag],
                max_severity="medium",
            )
        )

        return collection

    def test_exclude_flagged_pandas(
        self,
        sample_judgments_df: pd.DataFrame,
        collection_with_flags: AnalyticsCollection,
    ) -> None:
        """Test excluding flagged judgments with pandas."""
        # Only use first two rows to match collection
        df = sample_judgments_df.head(2)

        filtered = filter_flagged_judgments(
            df,
            collection_with_flags,
            exclude_flagged=True,
        )

        assert isinstance(filtered, pd.DataFrame)
        assert len(filtered) == 1
        assert filtered.iloc[0]["item_id"] == str(ITEM_IDS[0])

    def test_keep_flagged_only_pandas(
        self,
        sample_judgments_df: pd.DataFrame,
        collection_with_flags: AnalyticsCollection,
    ) -> None:
        """Test keeping only flagged judgments."""
        df = sample_judgments_df.head(2)

        filtered = filter_flagged_judgments(
            df,
            collection_with_flags,
            exclude_flagged=False,
        )

        assert len(filtered) == 1
        assert filtered.iloc[0]["item_id"] == str(ITEM_IDS[1])

    def test_exclude_flagged_polars(
        self,
        sample_judgments_df: pd.DataFrame,
        collection_with_flags: AnalyticsCollection,
    ) -> None:
        """Test excluding flagged judgments with polars."""
        df = pl.from_pandas(sample_judgments_df.head(2))

        filtered = filter_flagged_judgments(
            df,
            collection_with_flags,
            exclude_flagged=True,
        )

        assert isinstance(filtered, pl.DataFrame)
        assert len(filtered) == 1


class TestGetExclusionList:
    """Tests for get_exclusion_list function."""

    def test_exclusion_by_flag_rate(self) -> None:
        """Test identifying participants with high flag rates."""
        collection = AnalyticsCollection(name="test")

        # Participant with 50% flag rate
        flag = AnalysisFlag(
            type="test_flag",
            severity="medium",
            message="Test flag",
        )

        for i in range(4):
            collection = collection.with_analytics(
                JudgmentAnalytics(
                    item_id=uuid4(),
                    participant_id="p001",
                    trial_index=i,
                    session_id="s001",
                    response_time_ms=1000,
                    flags=[flag] if i < 2 else [],
                    max_severity="medium" if i < 2 else None,
                )
            )

        # Participant with 0% flag rate
        for i in range(4):
            collection = collection.with_analytics(
                JudgmentAnalytics(
                    item_id=uuid4(),
                    participant_id="p002",
                    trial_index=i,
                    session_id="s002",
                    response_time_ms=1500,
                )
            )

        # Exclude participants with >25% flag rate
        exclude = get_exclusion_list(collection, min_flag_rate=0.25)
        assert "p001" in exclude
        assert "p002" not in exclude

    def test_exclusion_by_severity(self) -> None:
        """Test exclusion filtering by severity."""
        collection = AnalyticsCollection(name="test")

        low_flag = AnalysisFlag(
            type="minor",
            severity="low",
            message="Minor issue",
        )

        high_flag = AnalysisFlag(
            type="major",
            severity="high",
            message="Major issue",
        )

        # Participant with only low-severity flags
        for i in range(4):
            collection = collection.with_analytics(
                JudgmentAnalytics(
                    item_id=uuid4(),
                    participant_id="p001",
                    trial_index=i,
                    session_id="s001",
                    response_time_ms=1000,
                    flags=[low_flag] if i < 2 else [],
                    max_severity="low" if i < 2 else None,
                )
            )

        # Participant with high-severity flags
        for i in range(4):
            collection = collection.with_analytics(
                JudgmentAnalytics(
                    item_id=uuid4(),
                    participant_id="p002",
                    trial_index=i,
                    session_id="s002",
                    response_time_ms=1000,
                    flags=[high_flag] if i < 2 else [],
                    max_severity="high" if i < 2 else None,
                )
            )

        # Only exclude based on high severity flags
        exclude = get_exclusion_list(
            collection, min_flag_rate=0.25, min_severity="high"
        )
        assert "p001" not in exclude  # Only has low severity
        assert "p002" in exclude  # Has high severity

    def test_empty_collection(self) -> None:
        """Test with empty collection."""
        collection = AnalyticsCollection(name="test")
        exclude = get_exclusion_list(collection)
        assert exclude == []
