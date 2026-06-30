"""Behavioral data extraction from slopit sessions.

This module provides functions for extracting per-judgment behavioral
analytics from slopit session data, using slopit's IO loaders and
analysis pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from slopit import load_session, load_sessions
from slopit.behavioral import (
    Analyzer,
    FocusAnalyzer,
    KeystrokeAnalyzer,
    PasteAnalyzer,
    TimingAnalyzer,
)
from slopit.pipeline import AnalysisPipeline
from slopit.schemas import AnalysisFlag, Severity, SlopitSession, SlopitTrial

from bead.behavioral.analytics import AnalyticsCollection, JudgmentAnalytics

if TYPE_CHECKING:
    from bead.data.base import JsonValue


def _get_max_severity(flags: list[AnalysisFlag]) -> Severity | None:
    """Get maximum severity from a list of flags.

    Parameters
    ----------
    flags : list[AnalysisFlag]
        List of analysis flags.

    Returns
    -------
    Severity | None
        Maximum severity, or None if no flags.
    """
    if not flags:
        return None

    severity_order: dict[str, int] = {"info": 0, "low": 1, "medium": 2, "high": 3}
    max_level = -1
    max_severity: Severity | None = None

    for flag in flags:
        level = severity_order.get(flag.severity, 0)
        if level > max_level:
            max_level = level
            max_severity = flag.severity

    return max_severity


def extract_from_trial(
    trial: SlopitTrial,
    session: SlopitSession,
    item_id_key: str = "item_id",
) -> JudgmentAnalytics | None:
    """Extract behavioral analytics from a single slopit trial.

    Parameters
    ----------
    trial : SlopitTrial
        Slopit trial data.
    session : SlopitSession
        Parent session for participant context.
    item_id_key : str
        Key in platform_data containing the item UUID.

    Returns
    -------
    JudgmentAnalytics | None
        Analytics record, or None if item_id not found in trial.
    """
    # Extract item_id from platform_data
    if trial.platform_data is None or item_id_key not in trial.platform_data:
        return None

    item_id_str = trial.platform_data[item_id_key]
    if not isinstance(item_id_str, str):
        return None

    try:
        item_id = UUID(item_id_str)
    except ValueError, TypeError:
        return None

    # Extract response value
    response_value: JsonValue = None
    if trial.response is not None:
        response_value = trial.response.value

    # Extract response time
    response_time_ms = trial.rt if trial.rt is not None else 0

    # Extract behavioral metrics
    keystroke_metrics = None
    focus_metrics = None
    timing_metrics = None
    paste_count = 0

    if trial.behavioral is not None:
        if trial.behavioral.metrics is not None:
            keystroke_metrics = trial.behavioral.metrics.keystroke
            focus_metrics = trial.behavioral.metrics.focus
            timing_metrics = trial.behavioral.metrics.timing

        if trial.behavioral.paste is not None:
            paste_count = len(trial.behavioral.paste)

    # Extract flags from capture_flags
    flags: list[AnalysisFlag] = []
    if trial.capture_flags is not None:
        # Convert CaptureFlags to AnalysisFlags for consistency
        for cf in trial.capture_flags:
            flags.append(
                AnalysisFlag(
                    type=cf.type,
                    analyzer="capture",
                    severity=cf.severity,
                    message=cf.message,
                    evidence=cf.details,
                    trial_ids=[trial.trial_id],
                )
            )

    return JudgmentAnalytics(
        item_id=item_id,
        participant_id=session.participant_id or session.session_id,
        trial_index=trial.trial_index,
        session_id=session.session_id,
        response_value=response_value,
        response_time_ms=response_time_ms,
        keystroke_metrics=keystroke_metrics,
        focus_metrics=focus_metrics,
        timing_metrics=timing_metrics,
        paste_event_count=paste_count,
        flags=flags,
        max_severity=_get_max_severity(flags),
    )


def extract_from_session(
    session: SlopitSession,
    item_id_key: str = "item_id",
) -> list[JudgmentAnalytics]:
    """Extract behavioral analytics from all trials in a slopit session.

    Parameters
    ----------
    session : SlopitSession
        Slopit session containing trial data.
    item_id_key : str
        Key in platform_data containing the item UUID.

    Returns
    -------
    list[JudgmentAnalytics]
        Analytics records for trials with valid item_id.
    """
    analytics: list[JudgmentAnalytics] = []

    for trial in session.trials:
        result = extract_from_trial(trial, session, item_id_key)
        if result is not None:
            analytics.append(result)

    return analytics


def extract_from_file(
    path: Path | str,
    item_id_key: str = "item_id",
) -> list[JudgmentAnalytics]:
    """Extract behavioral analytics from a slopit session file.

    Uses slopit's load_session() to automatically detect format.

    Parameters
    ----------
    path : Path | str
        Path to session file (JSON or JATOS format).
    item_id_key : str
        Key in platform_data containing the item UUID.

    Returns
    -------
    list[JudgmentAnalytics]
        Analytics records from the session.

    Examples
    --------
    >>> analytics = extract_from_file("data/session_001.json")
    >>> len(analytics)
    50
    """
    session = load_session(path)
    return extract_from_session(session, item_id_key)


def extract_from_directory(
    path: Path | str,
    pattern: str = "*",
    item_id_key: str = "item_id",
    name: str | None = None,
) -> AnalyticsCollection:
    """Extract behavioral analytics from all session files in a directory.

    Uses slopit's load_sessions() to load all files.

    Parameters
    ----------
    path : Path | str
        Directory containing session files.
    pattern : str
        Glob pattern for file matching (default: "*").
    item_id_key : str
        Key in platform_data containing the item UUID.
    name : str | None
        Name for the collection. Defaults to directory name.

    Returns
    -------
    AnalyticsCollection
        Collection of analytics from all sessions.

    Examples
    --------
    >>> collection = extract_from_directory("data/jatos_export/")
    >>> print(f"Extracted {len(collection)} analytics records")
    """
    path = Path(path)
    sessions = load_sessions(path, pattern)

    all_analytics: list[JudgmentAnalytics] = []
    for session in sessions:
        analytics = extract_from_session(session, item_id_key)
        all_analytics.extend(analytics)

    collection_name = name if name is not None else path.name
    return AnalyticsCollection(name=collection_name, analytics=all_analytics)


def analyze_sessions(
    sessions: list[SlopitSession],
    analyzers: list[Analyzer] | None = None,
) -> list[SlopitSession]:
    """Run slopit behavioral analyzers on sessions.

    Uses slopit's AnalysisPipeline to process sessions with
    the specified analyzers.

    Parameters
    ----------
    sessions : list[SlopitSession]
        Sessions to analyze.
    analyzers : list[Analyzer] | None
        Analyzers to run. If None, uses default set:
        KeystrokeAnalyzer, FocusAnalyzer, PasteAnalyzer, TimingAnalyzer.

    Returns
    -------
    list[SlopitSession]
        Sessions with analysis flags added.

    Examples
    --------
    >>> from slopit import load_sessions
    >>> sessions = load_sessions("data/")
    >>> analyzed = analyze_sessions(sessions)
    >>> # Sessions now have analysis flags populated
    """
    if analyzers is None:
        analyzers = [
            KeystrokeAnalyzer(),
            FocusAnalyzer(),
            PasteAnalyzer(),
            TimingAnalyzer(),
        ]

    pipeline = AnalysisPipeline(analyzers)
    return pipeline.analyze(sessions)


def extract_with_analysis(
    path: Path | str,
    pattern: str = "*",
    item_id_key: str = "item_id",
    analyzers: list[Analyzer] | None = None,
    name: str | None = None,
) -> AnalyticsCollection:
    """Load sessions, run analysis, and extract analytics in one step.

    Convenience function that combines loading, analysis, and extraction.

    Parameters
    ----------
    path : Path | str
        Path to session file or directory.
    pattern : str
        Glob pattern for directory (default: "*").
    item_id_key : str
        Key in platform_data containing the item UUID.
    analyzers : list[Analyzer] | None
        Analyzers to run. If None, uses default set.
    name : str | None
        Name for the collection.

    Returns
    -------
    AnalyticsCollection
        Collection with analyzed behavioral data.

    Examples
    --------
    >>> collection = extract_with_analysis("data/jatos_export/")
    >>> summaries = collection.get_participant_summaries()
    >>> for s in summaries:
    ...     if s.flag_rate > 0.1:
    ...         print(f"Participant {s.participant_id}: {s.flag_rate:.1%} flagged")
    """
    path = Path(path)

    # Load sessions
    sessions = load_sessions(path, pattern)

    # Run analysis
    analyzed = analyze_sessions(sessions, analyzers)

    # Extract analytics
    all_analytics: list[JudgmentAnalytics] = []
    for session in analyzed:
        analytics = extract_from_session(session, item_id_key)
        all_analytics.extend(analytics)

    collection_name = name if name is not None else path.name
    return AnalyticsCollection(name=collection_name, analytics=all_analytics)
