"""Timestamp preprocessing utilities for fixed-step feature generation v0.1.

This module prepares one symbol x one trading day CSV dataframe for the later
fixed-time-grid stage. It does not generate the fixed grid or trading features.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
import math
from typing import Any

import pandas as pd


ROW_TIMESTAMP_COLUMNS = (
    "BidTime",
    "AskTime",
    "CurrentPriceTime",
    "TradingVolumeTime",
)

PRESERVED_TIMESTAMP_COLUMNS = (
    "OpeningPriceTime",
    "HighPriceTime",
    "LowPriceTime",
    "PreviousCloseTime",
)

KNOWN_TIMESTAMP_COLUMNS = ROW_TIMESTAMP_COLUMNS + PRESERVED_TIMESTAMP_COLUMNS
TOKYO_TZ = "Asia/Tokyo"
TIMEZONE_SUFFIX_PATTERN = r"(?:Z|[+-]\d{2}:?\d{2})\s*$"


@dataclass
class TimestampPreprocessResult:
    df: pd.DataFrame
    qc: dict


def build_row_timestamp_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamp columns and add row_order + row_timestamp_raw.

    row_timestamp_raw is max(BidTime, AskTime, CurrentPriceTime,
    TradingVolumeTime). PreviousCloseTime, OpeningPriceTime, HighPriceTime, and
    LowPriceTime are intentionally excluded from the row timestamp source.
    """

    result = df.copy()
    result["row_order"] = range(len(result))

    for column in ROW_TIMESTAMP_COLUMNS:
        if column in result.columns:
            result[column] = _parse_timestamp_series_as_tokyo(result[column])
        else:
            result[column] = _empty_tokyo_timestamp_series(result.index)

    for column in PRESERVED_TIMESTAMP_COLUMNS:
        if column in result.columns:
            result[column] = _parse_timestamp_series_as_tokyo(result[column])

    row_timestamp_frame = pd.DataFrame(index=result.index)
    for column in ROW_TIMESTAMP_COLUMNS:
        row_timestamp_frame[column] = result[column]
    result["row_timestamp_raw"] = row_timestamp_frame.max(axis=1)

    return result


def filter_timestamp_reverse_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove rows where row_timestamp_raw goes backward versus prior row."""

    source = df if "row_timestamp_raw" in df.columns else build_row_timestamp_raw(df)
    result = source.copy()

    previous_timestamp = result["row_timestamp_raw"].shift(1)
    reverse_mask = (
        result["row_timestamp_raw"].notna()
        & previous_timestamp.notna()
        & (result["row_timestamp_raw"] < previous_timestamp)
    )

    qc = {
        "timestamp_reverse_rows_removed": int(reverse_mask.sum()),
    }

    return result.loc[~reverse_mask].copy(), qc


def filter_session_rows(
    df: pd.DataFrame,
    morning_start: str = "09:00:00",
    morning_end: str = "11:30:00",
    afternoon_start: str = "12:30:00",
    afternoon_end: str = "15:25:00",
) -> tuple[pd.DataFrame, dict]:
    """Keep only configured morning/afternoon session rows and label them."""

    source = df if "row_timestamp_raw" in df.columns else build_row_timestamp_raw(df)
    result = source.copy()

    morning_start_time = _parse_time(morning_start)
    morning_end_time = _parse_time(morning_end)
    afternoon_start_time = _parse_time(afternoon_start)
    afternoon_end_time = _parse_time(afternoon_end)

    timestamp_times = result["row_timestamp_raw"].map(_timestamp_time)

    pre_open_mask = _time_mask(
        timestamp_times, lambda value: value is not None and value < morning_start_time
    )
    lunch_break_mask = _time_mask(
        timestamp_times,
        lambda value: value is not None
        and morning_end_time <= value < afternoon_start_time,
    )
    morning_mask = _time_mask(
        timestamp_times,
        lambda value: value is not None
        and morning_start_time <= value < morning_end_time,
    )
    afternoon_mask = _time_mask(
        timestamp_times,
        lambda value: value is not None
        and afternoon_start_time <= value < afternoon_end_time,
    )
    keep_mask = morning_mask | afternoon_mask
    has_timestamp_mask = result["row_timestamp_raw"].notna()
    after_hours_mask = (
        has_timestamp_mask & ~keep_mask & ~pre_open_mask & ~lunch_break_mask
    )

    result["session"] = pd.NA
    result.loc[morning_mask, "session"] = "morning"
    result.loc[afternoon_mask, "session"] = "afternoon"
    filtered = result.loc[keep_mask].copy()

    qc = {
        "pre_open_rows_removed": int(pre_open_mask.sum()),
        "lunch_break_rows_removed": int(lunch_break_mask.sum()),
        "after_hours_rows_removed": int(after_hours_mask.sum()),
        "morning_rows": int((filtered["session"] == "morning").sum()),
        "afternoon_rows": int((filtered["session"] == "afternoon").sum()),
    }

    return filtered, qc


def add_episode_start_eligible_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add the provisional episode-start eligibility flag."""

    source = df if "row_timestamp_raw" in df.columns else build_row_timestamp_raw(df)
    result = source.copy()

    for column in ("CurrentPriceTime", "TradingVolumeTime"):
        if column in result.columns:
            result[column] = _parse_timestamp_series_as_tokyo(result[column])
        else:
            result[column] = _empty_tokyo_timestamp_series(result.index)

    timestamp_times = result["row_timestamp_raw"].map(_timestamp_time)
    after_morning_start = _time_mask(
        timestamp_times, lambda value: value is not None and value >= time(9, 0, 0)
    )
    has_trade_or_volume_time = (
        result["CurrentPriceTime"].notna() | result["TradingVolumeTime"].notna()
    )
    has_opening_price = _is_valid_value(result, "OpeningPrice")

    result["is_episode_start_eligible"] = (
        after_morning_start & has_trade_or_volume_time & has_opening_price
    )

    return result


def preprocess_timestamps_v01(df: pd.DataFrame) -> TimestampPreprocessResult:
    """Run timestamp v0.1 preprocessing and return dataframe plus QC log."""

    input_rows = len(df)
    timestamp_columns_missing = [
        column for column in KNOWN_TIMESTAMP_COLUMNS if column not in df.columns
    ]
    parse_failed_count_by_column = _timestamp_parse_failed_count_by_column(df)

    with_raw_timestamp = build_row_timestamp_raw(df)
    row_timestamp_missing_rows = int(with_raw_timestamp["row_timestamp_raw"].isna().sum())

    without_reverse, reverse_qc = filter_timestamp_reverse_rows(with_raw_timestamp)
    in_session, session_qc = filter_session_rows(without_reverse)
    output = add_episode_start_eligible_flag(in_session)

    qc: dict[str, Any] = {
        "input_rows": input_rows,
        "output_rows": len(output),
        "row_timestamp_missing_rows": row_timestamp_missing_rows,
        "timestamp_reverse_rows_removed": reverse_qc[
            "timestamp_reverse_rows_removed"
        ],
        "pre_open_rows_removed": session_qc["pre_open_rows_removed"],
        "lunch_break_rows_removed": session_qc["lunch_break_rows_removed"],
        "after_hours_rows_removed": session_qc["after_hours_rows_removed"],
        "morning_rows": session_qc["morning_rows"],
        "afternoon_rows": session_qc["afternoon_rows"],
        "episode_start_eligible_rows": int(
            output.get("is_episode_start_eligible", pd.Series(dtype=bool)).sum()
        ),
        "first_row_timestamp": _timestamp_to_iso(
            output["row_timestamp_raw"].iloc[0] if len(output) else pd.NaT
        ),
        "last_row_timestamp": _timestamp_to_iso(
            output["row_timestamp_raw"].iloc[-1] if len(output) else pd.NaT
        ),
        "timestamp_columns_missing": timestamp_columns_missing,
        "timestamp_parse_failed_count_by_column": parse_failed_count_by_column,
    }

    return TimestampPreprocessResult(df=output, qc=qc)


def _parse_time(value: str) -> time:
    return time.fromisoformat(value)


def _parse_timestamp_series_as_tokyo(series: pd.Series) -> pd.Series:
    """Parse timestamp values into Asia/Tokyo timestamps.

    Timezone-aware values are converted to Asia/Tokyo. Timezone-naive values are
    interpreted as Asia/Tokyo local times rather than UTC.
    """

    result = _empty_tokyo_timestamp_series(series.index)
    text = series.astype("string")
    has_value = series.notna() & text.str.strip().ne("")
    aware_mask = has_value & text.str.strip().str.contains(
        TIMEZONE_SUFFIX_PATTERN, regex=True, na=False
    )
    naive_mask = has_value & ~aware_mask

    if aware_mask.any():
        aware_values = pd.to_datetime(
            series.loc[aware_mask], errors="coerce", utc=True, format="mixed"
        ).dt.tz_convert(TOKYO_TZ)
        result.loc[aware_mask] = aware_values

    if naive_mask.any():
        naive_values = pd.to_datetime(
            series.loc[naive_mask], errors="coerce", format="mixed"
        )
        result.loc[naive_mask] = naive_values.dt.tz_localize(TOKYO_TZ)

    return result


def _empty_tokyo_timestamp_series(index: pd.Index) -> pd.Series:
    return pd.Series(pd.NaT, index=index, dtype=f"datetime64[ns, {TOKYO_TZ}]")


def _timestamp_time(value) -> time | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).time()


def _time_mask(timestamp_times: pd.Series, predicate) -> pd.Series:
    return pd.Series(
        [bool(predicate(value)) for value in timestamp_times],
        index=timestamp_times.index,
        dtype=bool,
    )


def _timestamp_to_iso(value) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _is_valid_value(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    numeric = pd.to_numeric(df[column], errors="coerce")
    return numeric.map(lambda value: math.isfinite(value) and value > 0)


def _timestamp_parse_failed_count_by_column(df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in KNOWN_TIMESTAMP_COLUMNS:
        if column not in df.columns:
            counts[column] = 0
            continue

        original = df[column]
        parsed = _parse_timestamp_series_as_tokyo(original)
        original_has_value = original.notna()
        if original.dtype == object:
            original_has_value = original_has_value & (original.astype(str).str.strip() != "")
        counts[column] = int((original_has_value & parsed.isna()).sum())

    return counts
