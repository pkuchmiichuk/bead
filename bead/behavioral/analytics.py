"""Behavioral analytics models.

Per-judgment behavioral metrics and participant-level summaries linking
slopit behavioral data to bead's item-based experimental structure.

The slopit metric classes (``KeystrokeMetrics``, ``FocusMetrics``,
``TimingMetrics``, ``AnalysisFlag``) are Pydantic models defined in an
upstream package; they appear here as ``dict[str, JsonValue]`` payloads
preserving the slopit field names. Consumers accessing them by attribute
should run ``slopit.schemas.KeystrokeMetrics.model_validate(payload)``
on the dict.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Literal, Self
from uuid import UUID

import didactic.api as dx
import pandas as pd
import polars as pl

from bead.data.base import BeadBaseModel, JsonValue
from bead.data.serialization import read_jsonlines, write_jsonlines

DataFrame = pd.DataFrame | pl.DataFrame
Severity = Literal["info", "low", "medium", "high"]


_SEVERITY_ORDER: dict[str, int] = {"info": 0, "low": 1, "medium": 2, "high": 3}


class AnalysisFlag(BeadBaseModel):
    """A single behavioral analysis flag.

    Attributes
    ----------
    type : str
        Flag identifier.
    severity : Severity
        Severity level.
    message : str | None
        Human-readable description.
    metadata : dict[str, JsonValue]
        Flag-specific metadata.
    """

    type: str
    severity: Severity
    message: str | None = None
    metadata: dict[str, JsonValue] = dx.field(default_factory=dict)


class JudgmentAnalytics(BeadBaseModel):
    """Behavioral analytics for a single judgment.

    Attributes
    ----------
    item_id : UUID
        Item being judged.
    participant_id : str
        Participant identifier.
    trial_index : int
        Position in the session (>= 0).
    session_id : str
        Slopit session identifier.
    response_value : JsonValue
        Participant's response value.
    response_time_ms : int
        Response time in milliseconds.
    keystroke_metrics : dict[str, JsonValue] | None
        Slopit keystroke-derived metrics.
    focus_metrics : dict[str, JsonValue] | None
        Slopit focus / visibility metrics.
    timing_metrics : dict[str, JsonValue] | None
        Slopit timing metrics.
    paste_event_count : int
        Number of paste events.
    flags : tuple[AnalysisFlag, ...]
        Behavioral flags from slopit analyzers.
    max_severity : Severity | None
        Maximum severity across flags.
    """

    __model_config__ = dx.ModelConfig(extra="ignore")

    item_id: UUID
    participant_id: str
    trial_index: int
    session_id: str
    response_time_ms: int
    response_value: JsonValue = None
    keystroke_metrics: dict[str, JsonValue] | None = None
    focus_metrics: dict[str, JsonValue] | None = None
    timing_metrics: dict[str, JsonValue] | None = None
    paste_event_count: int = 0
    flags: tuple[dx.Embed[AnalysisFlag], ...] = ()
    max_severity: Severity | None = None

    @property
    def has_paste_events(self) -> bool:
        """Whether the judgment had any paste events."""
        return self.paste_event_count > 0

    @property
    def is_flagged(self) -> bool:
        """Whether any analysis flags are present."""
        return len(self.flags) > 0

    def get_flag_types(self) -> tuple[str, ...]:
        """Return the flag-type identifiers."""
        return tuple(f.type for f in self.flags)


class ParticipantBehavioralSummary(BeadBaseModel):
    """Aggregated behavioral metrics for one participant.

    Attributes
    ----------
    participant_id : str
        Participant identifier.
    session_id : str
        Slopit session identifier.
    total_judgments : int
        Total judgments analyzed.
    flagged_judgments : int
        Number of judgments with at least one flag.
    mean_response_time_ms : float
        Mean response time in milliseconds.
    mean_iki : float | None
        Mean inter-keystroke interval.
    total_keystrokes : int
        Total keystrokes.
    total_paste_events : int
        Total paste events.
    total_blur_events : int
        Total window-blur events.
    total_blur_duration_ms : float
        Total time spent with the window blurred (ms).
    flag_counts : dict[str, int]
        Flag-type histogram.
    max_severity : Severity | None
        Maximum severity across the participant's flags.
    """

    __model_config__ = dx.ModelConfig(extra="ignore")

    participant_id: str
    session_id: str
    total_judgments: int
    mean_response_time_ms: float
    flagged_judgments: int = 0
    mean_iki: float | None = None
    total_keystrokes: int = 0
    total_paste_events: int = 0
    total_blur_events: int = 0
    total_blur_duration_ms: float = 0.0
    flag_counts: dict[str, int] = dx.field(default_factory=dict)
    max_severity: Severity | None = None

    @property
    def flag_rate(self) -> float:
        """Proportion of judgments that were flagged (0.0-1.0)."""
        if self.total_judgments == 0:
            return 0.0
        return self.flagged_judgments / self.total_judgments

    @property
    def has_paste_events(self) -> bool:
        """Whether the participant had any paste events."""
        return self.total_paste_events > 0


class AnalyticsCollection(BeadBaseModel):
    """Collection of judgment analytics.

    Attributes
    ----------
    name : str
        Collection name.
    analytics : tuple[JudgmentAnalytics, ...]
        Per-judgment records.
    """

    name: str
    analytics: tuple[dx.Embed[JudgmentAnalytics], ...] = ()

    def __len__(self) -> int:
        """Return the number of analytics records."""
        return len(self.analytics)

    def with_analytics(self, analytics: JudgmentAnalytics) -> Self:
        """Return a new collection with *analytics* appended."""
        return self.with_(analytics=(*self.analytics, analytics)).touched()

    def with_many(
        self,
        analytics_list: tuple[JudgmentAnalytics, ...] | list[JudgmentAnalytics],
    ) -> Self:
        """Return a new collection with each record appended."""
        return self.with_(analytics=(*self.analytics, *analytics_list)).touched()

    def get_by_participant(self, participant_id: str) -> tuple[JudgmentAnalytics, ...]:
        """Return analytics belonging to *participant_id*."""
        return tuple(a for a in self.analytics if a.participant_id == participant_id)

    def get_by_item(self, item_id: UUID) -> tuple[JudgmentAnalytics, ...]:
        """Return analytics for *item_id*."""
        return tuple(a for a in self.analytics if a.item_id == item_id)

    def filter_flagged(
        self,
        min_severity: Severity | None = None,
        exclude_flagged: bool = False,
    ) -> AnalyticsCollection:
        """Filter analytics by flag status.

        Parameters
        ----------
        min_severity
            Include only analytics with at least one flag at this severity
            or higher. Severity order is ``info < low < medium < high``.
        exclude_flagged
            If true, include only unflagged analytics.
        """

        def meets_criteria(record: JudgmentAnalytics) -> bool:
            has_flags = record.is_flagged
            if exclude_flagged:
                return not has_flags
            if not has_flags:
                return False
            if min_severity is None:
                return True
            min_level = _SEVERITY_ORDER.get(min_severity, 0)
            return any(
                _SEVERITY_ORDER.get(flag.severity, 0) >= min_level
                for flag in record.flags
            )

        filtered = tuple(a for a in self.analytics if meets_criteria(a))
        return AnalyticsCollection(name=f"{self.name}_filtered", analytics=filtered)

    def get_participant_ids(self) -> tuple[str, ...]:
        """Return the unique participant identifiers in the collection."""
        return tuple({a.participant_id for a in self.analytics})

    def get_participant_summaries(
        self,
    ) -> tuple[ParticipantBehavioralSummary, ...]:
        """Generate one ``ParticipantBehavioralSummary`` per participant."""
        by_participant: dict[str, list[JudgmentAnalytics]] = defaultdict(list)
        for a in self.analytics:
            by_participant[a.participant_id].append(a)

        summaries: list[ParticipantBehavioralSummary] = []
        for participant_id, records in by_participant.items():
            total = len(records)
            flagged = sum(1 for r in records if r.is_flagged)
            response_times = [r.response_time_ms for r in records]
            mean_rt = sum(response_times) / total if total > 0 else 0.0

            ikis: list[float] = []
            total_keystrokes = 0
            total_pastes = 0
            for r in records:
                total_pastes += r.paste_event_count
                if r.keystroke_metrics is not None:
                    total_keystrokes += int(
                        r.keystroke_metrics.get("total_keystrokes", 0) or 0
                    )
                    iki = r.keystroke_metrics.get("mean_iki")
                    if isinstance(iki, (int, float)) and iki > 0:
                        ikis.append(float(iki))

            mean_iki = sum(ikis) / len(ikis) if ikis else None

            blur_events = 0
            blur_duration = 0.0
            for r in records:
                if r.focus_metrics is not None:
                    blur_events += int(r.focus_metrics.get("blur_count", 0) or 0)
                    duration = r.focus_metrics.get("total_blur_duration", 0.0)
                    if isinstance(duration, (int, float)):
                        blur_duration += float(duration)

            flag_counts: dict[str, int] = defaultdict(int)
            max_severity_level = -1
            max_severity: Severity | None = None
            for r in records:
                for flag in r.flags:
                    flag_counts[flag.type] += 1
                    level = _SEVERITY_ORDER.get(flag.severity, 0)
                    if level > max_severity_level:
                        max_severity_level = level
                        max_severity = flag.severity

            session_id = records[0].session_id if records else ""
            summaries.append(
                ParticipantBehavioralSummary(
                    participant_id=participant_id,
                    session_id=session_id,
                    total_judgments=total,
                    flagged_judgments=flagged,
                    mean_response_time_ms=mean_rt,
                    mean_iki=mean_iki,
                    total_keystrokes=total_keystrokes,
                    total_paste_events=total_pastes,
                    total_blur_events=blur_events,
                    total_blur_duration_ms=blur_duration,
                    flag_counts=dict(flag_counts),
                    max_severity=max_severity,
                )
            )

        return tuple(summaries)

    def to_jsonl(self, path: Path | str) -> None:
        """Write analytics to *path* as JSONLines."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonlines(self.analytics, path)

    @classmethod
    def from_jsonl(
        cls,
        path: Path | str,
        name: str = "loaded_analytics",
    ) -> AnalyticsCollection:
        """Load analytics from a JSONLines file."""
        analytics = read_jsonlines(Path(path), JudgmentAnalytics)
        return cls(name=name, analytics=tuple(analytics))

    def to_dataframe(
        self,
        backend: Literal["pandas", "polars"] = "pandas",
        include_metrics: bool = True,
        include_flags: bool = True,
    ) -> DataFrame:
        """Render the collection as a pandas or polars DataFrame."""
        if not self.analytics:
            columns = [
                "item_id",
                "participant_id",
                "trial_index",
                "session_id",
                "response_value",
                "response_time_ms",
            ]
            if backend == "pandas":
                return pd.DataFrame(columns=columns)
            schema: dict[str, type[pl.Utf8]] = dict.fromkeys(columns, pl.Utf8)
            return pl.DataFrame(schema=schema)

        records: list[dict[str, JsonValue]] = []
        for a in self.analytics:
            record: dict[str, JsonValue] = {
                "item_id": str(a.item_id),
                "participant_id": a.participant_id,
                "trial_index": a.trial_index,
                "session_id": a.session_id,
                "response_value": a.response_value,
                "response_time_ms": a.response_time_ms,
                "paste_event_count": a.paste_event_count,
            }

            if include_metrics:
                if a.keystroke_metrics is not None:
                    record["keystroke_total"] = a.keystroke_metrics.get(
                        "total_keystrokes"
                    )
                    record["keystroke_mean_iki"] = a.keystroke_metrics.get("mean_iki")
                    record["keystroke_std_iki"] = a.keystroke_metrics.get("std_iki")
                    record["keystroke_deletions"] = a.keystroke_metrics.get("deletions")
                else:
                    record["keystroke_total"] = None
                    record["keystroke_mean_iki"] = None
                    record["keystroke_std_iki"] = None
                    record["keystroke_deletions"] = None

                if a.focus_metrics is not None:
                    record["focus_blur_count"] = a.focus_metrics.get("blur_count")
                    record["focus_blur_duration"] = a.focus_metrics.get(
                        "total_blur_duration"
                    )
                else:
                    record["focus_blur_count"] = None
                    record["focus_blur_duration"] = None

                if a.timing_metrics is not None:
                    record["timing_first_keystroke"] = a.timing_metrics.get(
                        "first_keystroke_latency"
                    )
                    record["timing_total_response"] = a.timing_metrics.get(
                        "total_response_time"
                    )
                else:
                    record["timing_first_keystroke"] = None
                    record["timing_total_response"] = None

            if include_flags:
                record["is_flagged"] = a.is_flagged
                record["flag_count"] = len(a.flags)
                record["max_severity"] = a.max_severity
                record["flag_types"] = ",".join(a.get_flag_types()) if a.flags else None

            records.append(record)

        if backend == "pandas":
            return pd.DataFrame(records)
        return pl.DataFrame(records)
