"""Past/current-only state-candidate feature generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


VOLATILITY_MOVEMENT_STATE_COLUMNS = (
    "realized_range_rate_30s",
    "realized_range_rate_120s",
    "realized_range_rate_300s",
    "abs_return_30s",
    "abs_return_120s",
    "abs_return_300s",
    "low_volatility_flag_30s",
    "low_volatility_flag_120s",
    "low_volatility_flag_300s",
    "high_volatility_flag_30s",
    "high_volatility_flag_120s",
    "high_volatility_flag_300s",
    "volatility_compression_flag",
    "recent_movement_insufficient_flag",
    "return_direction_persistence_30s",
    "return_direction_persistence_120s",
    "return_direction_persistence_300s",
    "range_expansion_ratio_30_300",
    "range_expansion_ratio_120_300",
)

SPREAD_LIQUIDITY_STATE_COLUMNS = (
    "spread_bps",
    "spread_zscore_by_symbol",
    "spread_regime_tight",
    "spread_regime_normal",
    "spread_regime_wide",
    "best_bid_ask_spread_abs",
    "top5_depth_buy_zscore",
    "top5_depth_sell_zscore",
    "top5_depth_min_side_zscore",
    "depth_regime_low",
    "depth_regime_normal",
    "depth_regime_high",
    "depth_imbalance",
    "depth_imbalance_abs",
    "buy_sell_top5_notional_ratio",
    "top5_slope_buy",
    "top5_slope_sell",
    "top5_slope_diff",
    "spread_to_realized_range_30s",
    "spread_to_realized_range_120s",
    "spread_to_realized_range_300s",
    "impact_cost_proxy_buy",
    "impact_cost_proxy_sell",
    "impact_cost_to_range_ratio_buy",
    "impact_cost_to_range_ratio_sell",
)

WINDOW_900S_STATE_COLUMNS = (
    "return_900s",
    "log_return_900s",
    "realized_range_rate_900s",
    "abs_return_900s",
    "volume_intensity_900s",
    "value_intensity_900s",
    "return_direction_persistence_900s",
    "range_position_900s",
    "mkt_1570_return_900s",
    "mkt_1570_log_return_900s",
    "mkt_1570_realized_range_rate_900s",
)


@dataclass(frozen=True)
class StateCandidateFeatureResult:
    df: pd.DataFrame
    qc: dict[str, Any]


def add_state_candidate_features_v01(
    df: pd.DataFrame,
    *,
    include_volatility_movement: bool = False,
    include_spread_liquidity: bool = False,
    include_900s_window: bool = False,
    step_seconds: int = 5,
) -> StateCandidateFeatureResult:
    """Add leakage-safe candidate features to a prepared episode dataframe."""

    output = df.copy()
    before_columns = set(output.columns)
    if include_volatility_movement or include_spread_liquidity or include_900s_window:
        output = _add_common_past_window_features(output, step_seconds=step_seconds)
    if include_volatility_movement:
        output = _add_volatility_movement_features(output)
    if include_spread_liquidity:
        output = _add_spread_liquidity_features(output)
    if include_900s_window:
        output = _add_900s_window_features(output, step_seconds=step_seconds)

    added_columns = [column for column in output.columns if column not in before_columns]
    candidate_columns = []
    if include_volatility_movement:
        candidate_columns.extend(VOLATILITY_MOVEMENT_STATE_COLUMNS)
    if include_spread_liquidity:
        candidate_columns.extend(SPREAD_LIQUIDITY_STATE_COLUMNS)
    if include_900s_window:
        candidate_columns.extend(WINDOW_900S_STATE_COLUMNS)
    candidate_columns = list(dict.fromkeys(candidate_columns))
    qc = {
        "added_columns": added_columns,
        "candidate_columns": candidate_columns,
        "candidate_column_count": len(candidate_columns),
        "feature_missing_count_by_column": {
            column: int(output[column].isna().sum())
            for column in candidate_columns
            if column in output.columns
        },
        "feature_inf_count_by_column": {
            column: _inf_count(output[column])
            for column in candidate_columns
            if column in output.columns
        },
        "future_information_used": False,
        "label_columns_used": False,
    }
    output.attrs["state_candidate_feature_qc"] = qc
    return StateCandidateFeatureResult(df=output, qc=qc)


def _add_common_past_window_features(df: pd.DataFrame, *, step_seconds: int) -> pd.DataFrame:
    out = df.copy()
    price = _num(out.get("CurrentPrice"))
    spread = _num(out.get("relative_spread"))
    for seconds in (30, 120, 300, 900):
        periods = max(1, int(seconds // step_seconds))
        lagged = _session_shift(out, price, periods)
        out[f"_candidate_lag_price_{seconds}s"] = lagged
        if seconds == 900:
            out["return_900s"] = _safe_div(price, lagged) - 1.0
            out["log_return_900s"] = np.log(_safe_div(price, lagged))
        high = _session_rolling(out, price, periods + 1, "max")
        low = _session_rolling(out, price, periods + 1, "min")
        out[f"realized_range_rate_{seconds}s"] = _safe_div(high - low, price)
        out[f"abs_return_{seconds}s"] = (
            (_safe_div(price, lagged) - 1.0).abs()
            if seconds == 900
            else _num(out.get(f"return_{seconds}s")).abs()
        )
        out[f"return_direction_persistence_{seconds}s"] = _direction_persistence(
            out, "return_5s", seconds, step_seconds=step_seconds
        )
        out[f"_range_low_{seconds}s"] = low
        out[f"_range_high_{seconds}s"] = high

    rr30 = _num(out.get("realized_range_rate_30s"))
    rr120 = _num(out.get("realized_range_rate_120s"))
    rr300 = _num(out.get("realized_range_rate_300s"))
    threshold_base = np.maximum(spread.fillna(0.0).abs() * 3.0, 0.0005)
    for seconds in (30, 120, 300):
        rr = _num(out.get(f"realized_range_rate_{seconds}s"))
        out[f"low_volatility_flag_{seconds}s"] = rr.le(threshold_base).fillna(False).astype(float)
        out[f"high_volatility_flag_{seconds}s"] = rr.ge(np.maximum(threshold_base * 4.0, 0.002)).fillna(False).astype(float)
    out["volatility_compression_flag"] = (
        rr30.lt(rr300 * 0.5) & rr300.gt(threshold_base * 2.0)
    ).fillna(False).astype(float)
    out["recent_movement_insufficient_flag"] = (
        _num(out.get("abs_return_300s")).le(threshold_base)
    ).fillna(False).astype(float)
    out["range_expansion_ratio_30_300"] = _safe_div(rr30, rr300)
    out["range_expansion_ratio_120_300"] = _safe_div(rr120, rr300)
    return out


def _add_volatility_movement_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def _add_spread_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    spread_abs = _num(out.get("spread"))
    relative_spread = _num(out.get("relative_spread"))
    out["spread_bps"] = relative_spread * 10000.0
    out["best_bid_ask_spread_abs"] = spread_abs.abs()
    out["spread_zscore_by_symbol"] = _past_current_zscore(out, out["spread_bps"])

    spread_regime = _three_regime_by_past_expanding(out, out["spread_bps"], lower_is_low=True)
    out["spread_regime_tight"] = spread_regime.eq("low").astype(float)
    out["spread_regime_normal"] = spread_regime.eq("normal").astype(float)
    out["spread_regime_wide"] = spread_regime.eq("high").astype(float)

    buy_depth = _num(out.get("buy_qty_top5_sum"))
    sell_depth = _num(out.get("sell_qty_top5_sum"))
    min_depth = pd.concat([buy_depth, sell_depth], axis=1).min(axis=1)
    out["top5_depth_buy_zscore"] = _past_current_zscore(out, buy_depth)
    out["top5_depth_sell_zscore"] = _past_current_zscore(out, sell_depth)
    out["top5_depth_min_side_zscore"] = _past_current_zscore(out, min_depth)
    depth_regime = _three_regime_by_past_expanding(out, min_depth, lower_is_low=True)
    out["depth_regime_low"] = depth_regime.eq("low").astype(float)
    out["depth_regime_normal"] = depth_regime.eq("normal").astype(float)
    out["depth_regime_high"] = depth_regime.eq("high").astype(float)

    out["depth_imbalance"] = _safe_div(buy_depth - sell_depth, buy_depth + sell_depth)
    out["depth_imbalance_abs"] = out["depth_imbalance"].abs()
    buy_notional = _top5_notional(out, "Buy")
    sell_notional = _top5_notional(out, "Sell")
    out["buy_sell_top5_notional_ratio"] = _safe_div(buy_notional, sell_notional)
    out["top5_slope_buy"] = _safe_div(
        _num(out.get("Buy1_Price")) - _num(out.get("Buy5_Price")),
        _num(out.get("Buy1_Price")),
    )
    out["top5_slope_sell"] = _safe_div(
        _num(out.get("Sell5_Price")) - _num(out.get("Sell1_Price")),
        _num(out.get("Sell1_Price")),
    )
    out["top5_slope_diff"] = out["top5_slope_buy"] - out["top5_slope_sell"]

    for seconds in (30, 120, 300):
        rr = _num(out.get(f"realized_range_rate_{seconds}s"))
        out[f"spread_to_realized_range_{seconds}s"] = _safe_div(relative_spread.abs(), rr)

    mid = _num(out.get("mid_price"))
    weighted_sell = _weighted_top5_price(out, "Sell")
    weighted_buy = _weighted_top5_price(out, "Buy")
    out["impact_cost_proxy_buy"] = _safe_div(weighted_sell - _num(out.get("best_sell_price")), mid)
    out["impact_cost_proxy_sell"] = _safe_div(_num(out.get("best_buy_price")) - weighted_buy, mid)
    rr300 = _num(out.get("realized_range_rate_300s"))
    out["impact_cost_to_range_ratio_buy"] = _safe_div(out["impact_cost_proxy_buy"].abs(), rr300)
    out["impact_cost_to_range_ratio_sell"] = _safe_div(out["impact_cost_proxy_sell"].abs(), rr300)
    return out


def _add_900s_window_features(df: pd.DataFrame, *, step_seconds: int) -> pd.DataFrame:
    out = df.copy()
    price = _num(out.get("CurrentPrice"))
    low = _num(out.get("_range_low_900s"))
    high = _num(out.get("_range_high_900s"))
    out["range_position_900s"] = _safe_div(price - low, high - low)
    volume = _num(out.get("TradingVolume"))
    value = _num(out.get("TradingValue"))
    periods = max(1, int(900 // step_seconds))
    out["volume_intensity_900s"] = (volume - _session_shift(out, volume, periods)).mask(lambda s: s < 0) / 900.0
    out["value_intensity_900s"] = (value - _session_shift(out, value, periods)).mask(lambda s: s < 0) / 900.0
    out["mkt_1570_return_900s"] = np.nan
    out["mkt_1570_log_return_900s"] = np.nan
    out["mkt_1570_realized_range_rate_900s"] = np.nan
    return out


def _session_shift(df: pd.DataFrame, series: pd.Series, periods: int) -> pd.Series:
    if "session" not in df.columns:
        return series.shift(periods)
    return series.groupby(df["session"], sort=False).shift(periods)


def _session_rolling(
    df: pd.DataFrame,
    series: pd.Series,
    window: int,
    op: str,
) -> pd.Series:
    if "session" not in df.columns:
        grouped = [(None, series)]
    else:
        grouped = series.groupby(df["session"], sort=False)
    if op == "max":
        return grouped.transform(lambda s: s.rolling(window=window, min_periods=1).max())
    if op == "min":
        return grouped.transform(lambda s: s.rolling(window=window, min_periods=1).min())
    raise ValueError(f"unsupported rolling op: {op}")


def _direction_persistence(
    df: pd.DataFrame,
    return_column: str,
    seconds: int,
    *,
    step_seconds: int,
) -> pd.Series:
    returns = _num(df.get(return_column))
    signs = np.sign(returns).replace(0.0, np.nan)
    window = max(1, int(seconds // step_seconds))
    if "session" in df.columns:
        rolling = signs.groupby(df["session"], sort=False).transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
    else:
        rolling = signs.rolling(window=window, min_periods=1).mean()
    return rolling.abs()


def _past_current_zscore(df: pd.DataFrame, series: pd.Series) -> pd.Series:
    values = _num(series)
    if "session" in df.columns:
        mean = values.groupby(df["session"], sort=False).transform(
            lambda s: s.expanding(min_periods=5).mean()
        )
        std = values.groupby(df["session"], sort=False).transform(
            lambda s: s.expanding(min_periods=5).std()
        )
    else:
        mean = values.expanding(min_periods=5).mean()
        std = values.expanding(min_periods=5).std()
    return _safe_div(values - mean, std)


def _three_regime_by_past_expanding(
    df: pd.DataFrame,
    series: pd.Series,
    *,
    lower_is_low: bool,
) -> pd.Series:
    values = _num(series)

    def classify(group: pd.Series) -> pd.Series:
        expanding = group.expanding(min_periods=10)
        low = expanding.quantile(0.25)
        high = expanding.quantile(0.75)
        result = pd.Series("normal", index=group.index, dtype="object")
        result[group <= low] = "low" if lower_is_low else "high"
        result[group >= high] = "high" if lower_is_low else "low"
        result[group.isna() | low.isna() | high.isna()] = "normal"
        return result

    if "session" in df.columns:
        return values.groupby(df["session"], sort=False, group_keys=False).apply(classify)
    return classify(values)


def _top5_notional(df: pd.DataFrame, side: str) -> pd.Series:
    total = pd.Series(0.0, index=df.index, dtype="float64")
    valid = pd.Series(True, index=df.index)
    for level in range(1, 6):
        price = _num(df.get(f"{side}{level}_Price"))
        qty = _num(df.get(f"{side}{level}_Qty"))
        total = total + price * qty
        valid &= np.isfinite(price) & np.isfinite(qty) & (price > 0) & (qty >= 0)
    return total.mask(~valid)


def _weighted_top5_price(df: pd.DataFrame, side: str) -> pd.Series:
    notional = _top5_notional(df, side)
    qty_total = pd.Series(0.0, index=df.index, dtype="float64")
    valid = pd.Series(True, index=df.index)
    for level in range(1, 6):
        qty = _num(df.get(f"{side}{level}_Qty"))
        qty_total = qty_total + qty
        valid &= np.isfinite(qty) & (qty >= 0)
    return _safe_div(notional, qty_total).mask(~valid)


def _num(value: Any) -> pd.Series:
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce")
    return pd.Series(dtype="float64")


def _safe_div(numerator: Any, denominator: Any) -> pd.Series:
    num = _num(numerator)
    den = _num(denominator)
    result = num / den
    return result.mask(~np.isfinite(result) | den.eq(0))


def _inf_count(series: pd.Series) -> int:
    numeric = pd.to_numeric(series, errors="coerce")
    return int(np.isinf(numeric.to_numpy(dtype="float64")).sum())

