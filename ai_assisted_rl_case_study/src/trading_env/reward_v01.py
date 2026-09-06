"""Reward v0.1 calculation utilities.

The module is intentionally independent from a full TradingEnv implementation.
All calculations are based on the provided Top5 order book snapshot and fixed
100-share position size.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Optional


Position = Literal["Flat", "Long", "Short"]
Action = Literal["Buy", "Sell", "Hold"]
BookSide = Literal["SellTop5", "BuyTop5", "none"]

ORDER_QTY = 100
NAN = float("nan")


@dataclass(frozen=True)
class OrderBookLevel:
    price: Optional[float]
    qty: Optional[float]


@dataclass(frozen=True)
class Top5ExecutionResult:
    executable: bool
    price: Optional[float]
    filled_qty: int
    available_qty: float
    used_book_side: BookSide
    failure_reason: Optional[str]


@dataclass(frozen=True)
class ReferenceNotionalResult:
    valid: bool
    reference_notional: Optional[float]
    failure_reason: Optional[str]


@dataclass(frozen=True)
class ExecutionUpdateResult:
    position_after: Position
    cash_pnl_after: float
    executed: bool
    execution_failed: bool
    execution_price: Optional[float]
    execution_qty: int
    used_book_side: BookSide
    invalid_action: bool
    failure_reason: Optional[str]


@dataclass(frozen=True)
class ValuationResult:
    equity: Optional[float]
    liquidation_price: Optional[float]
    valuation_failed: bool
    used_book_side: BookSide
    top5_available_qty: float
    failure_reason: Optional[str]


@dataclass(frozen=True)
class RewardResult:
    reward_raw_yen: Optional[float]
    reward: Optional[float]
    reward_valid: bool
    failure_reason: Optional[str]


@dataclass(frozen=True)
class StepRewardResult:
    position_after: Position
    cash_pnl_after: float
    equity_after: Optional[float]
    reward_raw_yen: Optional[float]
    reward: Optional[float]
    reward_valid: bool
    training_valid: bool
    executed: bool
    execution_failed: bool
    execution_price: Optional[float]
    execution_qty: int
    invalid_action: bool
    valuation_failed: bool
    is_forced_exit: bool
    forced_exit_failed: bool
    execution_used_book_side: BookSide
    valuation_used_book_side: BookSide
    failure_reason: Optional[str]
    reference_notional: Optional[float] = None
    review_required: bool = False
    reward_adjustment_raw_yen: float = 0.0
    loss_delay_penalty_applied: bool = False
    loss_delay_penalty_raw_yen: float = 0.0
    loss_delay_unrealized_loss_threshold_yen: Optional[float] = None
    loss_delay_unrealized_pnl_yen: Optional[float] = None
    loss_delay_close_action: Optional[Action] = None
    loss_delay_close_executable: bool = False
    loss_delay_loss_state_steps: int = 0
    loss_delay_min_persistence_steps: int = 1
    loss_delay_holding_time_seconds: float = 0.0
    loss_delay_min_holding_time_seconds: float = 0.0
    loss_delay_recent_unrealized_pnl_yen: Optional[float] = None
    loss_delay_no_recent_improvement: bool = False
    loss_delay_penalty_cap_yen: float = 0.0
    loss_delay_penalty_cap_remaining_yen: Optional[float] = None
    new_entry_penalty_applied: bool = False
    new_entry_penalty_raw_yen: float = 0.0
    new_entry_penalty_apply_to: str = "off"
    new_entry_penalty_entry_side: Optional[str] = None


def validate_reference_notional(previous_close: float) -> ReferenceNotionalResult:
    if previous_close is None:
        return ReferenceNotionalResult(False, None, "invalid_previous_close")

    try:
        value = float(previous_close)
    except (TypeError, ValueError):
        return ReferenceNotionalResult(False, None, "invalid_previous_close")

    if not math.isfinite(value) or value <= 0:
        return ReferenceNotionalResult(False, None, "invalid_previous_close")

    return ReferenceNotionalResult(True, ORDER_QTY * value, None)


def compute_top5_weighted_price(
    levels: list[OrderBookLevel],
    qty: int = ORDER_QTY,
    used_book_side: BookSide = "none",
) -> Top5ExecutionResult:
    if qty <= 0:
        return Top5ExecutionResult(
            executable=False,
            price=None,
            filled_qty=0,
            available_qty=0.0,
            used_book_side=used_book_side,
            failure_reason="invalid_order_qty",
        )

    remaining_qty = float(qty)
    notional = 0.0
    available_qty = 0.0

    for level in levels:
        if level.price is None or level.qty is None:
            return Top5ExecutionResult(
                executable=False,
                price=None,
                filled_qty=0,
                available_qty=available_qty,
                used_book_side=used_book_side,
                failure_reason="missing_order_book",
            )

        try:
            price = float(level.price)
            level_qty = float(level.qty)
        except (TypeError, ValueError):
            return Top5ExecutionResult(
                executable=False,
                price=None,
                filled_qty=0,
                available_qty=available_qty,
                used_book_side=used_book_side,
                failure_reason="missing_order_book",
            )

        if not math.isfinite(price) or price <= 0:
            return Top5ExecutionResult(
                executable=False,
                price=None,
                filled_qty=0,
                available_qty=available_qty,
                used_book_side=used_book_side,
                failure_reason="invalid_order_book_price",
            )

        if not math.isfinite(level_qty) or level_qty < 0:
            return Top5ExecutionResult(
                executable=False,
                price=None,
                filled_qty=0,
                available_qty=available_qty,
                used_book_side=used_book_side,
                failure_reason="invalid_order_book_qty",
            )

        available_qty += level_qty
        fill_qty = min(remaining_qty, level_qty)
        notional += fill_qty * price
        remaining_qty -= fill_qty

        if remaining_qty <= 0:
            return Top5ExecutionResult(
                executable=True,
                price=notional / qty,
                filled_qty=qty,
                available_qty=available_qty,
                used_book_side=used_book_side,
                failure_reason=None,
            )

    return Top5ExecutionResult(
        executable=False,
        price=None,
        filled_qty=0,
        available_qty=available_qty,
        used_book_side=used_book_side,
        failure_reason="top5_liquidity_shortage",
    )


def is_valid_action(
    position_before: Position,
    action: Action,
) -> bool:
    valid_actions: dict[Position, set[Action]] = {
        "Flat": {"Buy", "Sell", "Hold"},
        "Long": {"Sell", "Hold"},
        "Short": {"Buy", "Hold"},
    }
    return action in valid_actions.get(position_before, set())


def apply_action(
    position_before: Position,
    cash_pnl_before: float,
    action: Action,
    order_book: dict,
) -> ExecutionUpdateResult:
    if not is_valid_action(position_before, action):
        return ExecutionUpdateResult(
            position_after=position_before,
            cash_pnl_after=cash_pnl_before,
            executed=False,
            execution_failed=False,
            execution_price=None,
            execution_qty=0,
            used_book_side="none",
            invalid_action=True,
            failure_reason="invalid_action",
        )

    if action == "Hold":
        return ExecutionUpdateResult(
            position_after=position_before,
            cash_pnl_after=cash_pnl_before,
            executed=False,
            execution_failed=False,
            execution_price=None,
            execution_qty=0,
            used_book_side="none",
            invalid_action=False,
            failure_reason=None,
        )

    used_book_side: BookSide = "SellTop5" if action == "Buy" else "BuyTop5"
    execution = compute_top5_weighted_price(
        _extract_top5_levels(order_book, used_book_side),
        qty=ORDER_QTY,
        used_book_side=used_book_side,
    )

    if not execution.executable or execution.price is None:
        return ExecutionUpdateResult(
            position_after=position_before,
            cash_pnl_after=cash_pnl_before,
            executed=False,
            execution_failed=True,
            execution_price=None,
            execution_qty=0,
            used_book_side=used_book_side,
            invalid_action=False,
            failure_reason=execution.failure_reason,
        )

    cash_delta = ORDER_QTY * execution.price

    if action == "Buy":
        position_after: Position = "Long" if position_before == "Flat" else "Flat"
        cash_pnl_after = cash_pnl_before - cash_delta
    else:
        position_after = "Short" if position_before == "Flat" else "Flat"
        cash_pnl_after = cash_pnl_before + cash_delta

    return ExecutionUpdateResult(
        position_after=position_after,
        cash_pnl_after=cash_pnl_after,
        executed=True,
        execution_failed=False,
        execution_price=execution.price,
        execution_qty=execution.filled_qty,
        used_book_side=used_book_side,
        invalid_action=False,
        failure_reason=None,
    )


def compute_liquidation_equity(
    position: Position,
    cash_pnl: float,
    order_book: dict,
) -> ValuationResult:
    try:
        cash_value = float(cash_pnl)
    except (TypeError, ValueError):
        return ValuationResult(None, None, True, "none", 0.0, "invalid_cash_pnl")

    if not math.isfinite(cash_value):
        return ValuationResult(None, None, True, "none", 0.0, "invalid_cash_pnl")

    if position == "Flat":
        return ValuationResult(cash_value, None, False, "none", 0.0, None)

    if position == "Long":
        used_book_side: BookSide = "BuyTop5"
        sign = 1.0
    elif position == "Short":
        used_book_side = "SellTop5"
        sign = -1.0
    else:
        return ValuationResult(None, None, True, "none", 0.0, "invalid_position")

    execution = compute_top5_weighted_price(
        _extract_top5_levels(order_book, used_book_side),
        qty=ORDER_QTY,
        used_book_side=used_book_side,
    )

    if not execution.executable or execution.price is None:
        return ValuationResult(
            equity=None,
            liquidation_price=None,
            valuation_failed=True,
            used_book_side=used_book_side,
            top5_available_qty=execution.available_qty,
            failure_reason=execution.failure_reason,
        )

    equity = cash_value + sign * ORDER_QTY * execution.price
    return ValuationResult(
        equity=equity,
        liquidation_price=execution.price,
        valuation_failed=False,
        used_book_side=used_book_side,
        top5_available_qty=execution.available_qty,
        failure_reason=None,
    )


def compute_reward(
    equity_before: float,
    equity_after: float,
    reference_notional: float,
) -> RewardResult:
    try:
        before = float(equity_before)
        after = float(equity_after)
        reference = float(reference_notional)
    except (TypeError, ValueError):
        return RewardResult(None, NAN, False, "invalid_reward_input")

    if not all(math.isfinite(value) for value in (before, after, reference)):
        return RewardResult(None, NAN, False, "invalid_reward_input")

    if reference <= 0:
        return RewardResult(None, NAN, False, "invalid_reference_notional")

    reward_raw_yen = after - before
    return RewardResult(
        reward_raw_yen=reward_raw_yen,
        reward=reward_raw_yen / reference,
        reward_valid=True,
        failure_reason=None,
    )


def handle_forced_exit(
    position_before: Position,
    cash_pnl_before: float,
    order_book: dict,
) -> ExecutionUpdateResult:
    if position_before == "Flat":
        return ExecutionUpdateResult(
            position_after="Flat",
            cash_pnl_after=cash_pnl_before,
            executed=False,
            execution_failed=False,
            execution_price=None,
            execution_qty=0,
            used_book_side="none",
            invalid_action=False,
            failure_reason=None,
        )

    action: Action = "Sell" if position_before == "Long" else "Buy"
    return apply_action(position_before, cash_pnl_before, action, order_book)


def step_reward_v01(
    position_before: Position,
    cash_pnl_before: float,
    equity_before: float,
    action: Action,
    order_book: dict,
    previous_close: float,
    is_forced_exit: bool = False,
    loss_delay_penalty_raw_yen: float = 0.0,
    loss_delay_unrealized_loss_threshold_yen: float = 50_000.0,
    loss_delay_require_close_executable: bool = True,
    loss_delay_entry_cash_pnl_before: Optional[float] = None,
    loss_delay_loss_state_steps_before: int = 0,
    loss_delay_min_persistence_steps: int = 1,
    loss_delay_holding_time_seconds: float = 0.0,
    loss_delay_min_holding_time_seconds: float = 0.0,
    loss_delay_require_no_recent_loss_improvement: bool = False,
    loss_delay_recent_unrealized_pnl_yen: Optional[float] = None,
    loss_delay_roundtrip_penalty_accrued_yen: float = 0.0,
    loss_delay_per_roundtrip_cap_yen: float = 0.0,
    new_entry_penalty_raw_yen: float = 0.0,
    new_entry_penalty_apply_to: str = "off",
) -> StepRewardResult:
    reference = validate_reference_notional(previous_close)

    if is_forced_exit:
        execution_update = handle_forced_exit(
            position_before, cash_pnl_before, order_book
        )
        forced_exit_failed = (
            position_before != "Flat" and execution_update.execution_failed
        )

        if forced_exit_failed:
            return StepRewardResult(
                position_after=execution_update.position_after,
                cash_pnl_after=execution_update.cash_pnl_after,
                equity_after=None,
                reward_raw_yen=None,
                reward=NAN,
                reward_valid=False,
                training_valid=False,
                executed=False,
                execution_failed=True,
                execution_price=None,
                execution_qty=0,
                invalid_action=execution_update.invalid_action,
                valuation_failed=False,
                is_forced_exit=True,
                forced_exit_failed=True,
                execution_used_book_side=execution_update.used_book_side,
                valuation_used_book_side="none",
                failure_reason=(
                    f"forced_exit_failed:{execution_update.failure_reason}"
                    if execution_update.failure_reason
                    else "forced_exit_failed"
                ),
                reference_notional=reference.reference_notional,
                review_required=True,
            )
    else:
        execution_update = apply_action(
            position_before, cash_pnl_before, action, order_book
        )

    valuation = compute_liquidation_equity(
        execution_update.position_after,
        execution_update.cash_pnl_after,
        order_book,
    )

    if valuation.valuation_failed:
        return StepRewardResult(
            position_after=execution_update.position_after,
            cash_pnl_after=execution_update.cash_pnl_after,
            equity_after=None,
            reward_raw_yen=None,
            reward=NAN,
            reward_valid=False,
            training_valid=reference.valid,
            executed=execution_update.executed,
            execution_failed=execution_update.execution_failed,
            execution_price=execution_update.execution_price,
            execution_qty=execution_update.execution_qty,
            invalid_action=execution_update.invalid_action,
            valuation_failed=True,
            is_forced_exit=is_forced_exit,
            forced_exit_failed=False,
            execution_used_book_side=execution_update.used_book_side,
            valuation_used_book_side=valuation.used_book_side,
            failure_reason=(
                valuation.failure_reason
                if reference.valid
                else _join_failure_reasons(
                    reference.failure_reason, valuation.failure_reason
                )
            ),
            reference_notional=reference.reference_notional,
            review_required=not reference.valid,
        )

    if not reference.valid or reference.reference_notional is None:
        return StepRewardResult(
            position_after=execution_update.position_after,
            cash_pnl_after=execution_update.cash_pnl_after,
            equity_after=valuation.equity,
            reward_raw_yen=None,
            reward=NAN,
            reward_valid=False,
            training_valid=False,
            executed=execution_update.executed,
            execution_failed=execution_update.execution_failed,
            execution_price=execution_update.execution_price,
            execution_qty=execution_update.execution_qty,
            invalid_action=execution_update.invalid_action,
            valuation_failed=False,
            is_forced_exit=is_forced_exit,
            forced_exit_failed=False,
            execution_used_book_side=execution_update.used_book_side,
            valuation_used_book_side=valuation.used_book_side,
            failure_reason=reference.failure_reason,
            reference_notional=None,
            review_required=True,
        )

    reward_result = compute_reward(
        equity_before, valuation.equity, reference.reference_notional
    )
    loss_delay_penalty = _compute_loss_delay_penalty(
        position_before=position_before,
        action=action,
        order_book=order_book,
        equity_after=valuation.equity,
        is_forced_exit=is_forced_exit,
        penalty_raw_yen=loss_delay_penalty_raw_yen,
        unrealized_loss_threshold_yen=loss_delay_unrealized_loss_threshold_yen,
        require_close_executable=loss_delay_require_close_executable,
        entry_cash_pnl_before=loss_delay_entry_cash_pnl_before,
        loss_state_steps_before=loss_delay_loss_state_steps_before,
        min_persistence_steps=loss_delay_min_persistence_steps,
        holding_time_seconds=loss_delay_holding_time_seconds,
        min_holding_time_seconds=loss_delay_min_holding_time_seconds,
        require_no_recent_loss_improvement=(
            loss_delay_require_no_recent_loss_improvement
        ),
        recent_unrealized_pnl_yen=loss_delay_recent_unrealized_pnl_yen,
        roundtrip_penalty_accrued_yen=loss_delay_roundtrip_penalty_accrued_yen,
        per_roundtrip_cap_yen=loss_delay_per_roundtrip_cap_yen,
    )
    new_entry_penalty = _compute_new_entry_penalty(
        position_before=position_before,
        position_after=execution_update.position_after,
        action=action,
        executed=execution_update.executed,
        execution_failed=execution_update.execution_failed,
        is_forced_exit=is_forced_exit,
        penalty_raw_yen=new_entry_penalty_raw_yen,
        apply_to=new_entry_penalty_apply_to,
    )
    reward_raw_yen = reward_result.reward_raw_yen
    reward = reward_result.reward
    if (
        reward_result.reward_valid
        and reward_result.reward_raw_yen is not None
    ):
        adjustment = float(loss_delay_penalty["penalty_raw_yen"]) + float(
            new_entry_penalty["penalty_raw_yen"]
        )
        if adjustment > 0.0:
            reward_raw_yen = float(reward_result.reward_raw_yen) - adjustment
            reward = reward_raw_yen / float(reference.reference_notional)

    return StepRewardResult(
        position_after=execution_update.position_after,
        cash_pnl_after=execution_update.cash_pnl_after,
        equity_after=valuation.equity,
        reward_raw_yen=reward_raw_yen,
        reward=reward,
        reward_valid=reward_result.reward_valid,
        training_valid=reward_result.reward_valid,
        executed=execution_update.executed,
        execution_failed=execution_update.execution_failed,
        execution_price=execution_update.execution_price,
        execution_qty=execution_update.execution_qty,
        invalid_action=execution_update.invalid_action,
        valuation_failed=False,
        is_forced_exit=is_forced_exit,
        forced_exit_failed=False,
        execution_used_book_side=execution_update.used_book_side,
        valuation_used_book_side=valuation.used_book_side,
        failure_reason=(
            reward_result.failure_reason
            or execution_update.failure_reason
            or valuation.failure_reason
        ),
        reference_notional=reference.reference_notional,
        review_required=not reward_result.reward_valid,
        reward_adjustment_raw_yen=-(
            float(loss_delay_penalty["penalty_raw_yen"])
            + float(new_entry_penalty["penalty_raw_yen"])
        ),
        loss_delay_penalty_applied=bool(loss_delay_penalty["applied"]),
        loss_delay_penalty_raw_yen=float(loss_delay_penalty["penalty_raw_yen"])
        if loss_delay_penalty["applied"]
        else 0.0,
        loss_delay_unrealized_loss_threshold_yen=float(
            loss_delay_penalty["threshold_yen"]
        )
        if loss_delay_penalty["threshold_yen"] is not None
        else None,
        loss_delay_unrealized_pnl_yen=loss_delay_penalty["unrealized_pnl_yen"],
        loss_delay_close_action=loss_delay_penalty["close_action"],
        loss_delay_close_executable=bool(loss_delay_penalty["close_executable"]),
        loss_delay_loss_state_steps=int(loss_delay_penalty["loss_state_steps"]),
        loss_delay_min_persistence_steps=int(
            loss_delay_penalty["min_persistence_steps"]
        ),
        loss_delay_holding_time_seconds=float(
            loss_delay_penalty["holding_time_seconds"]
        ),
        loss_delay_min_holding_time_seconds=float(
            loss_delay_penalty["min_holding_time_seconds"]
        ),
        loss_delay_recent_unrealized_pnl_yen=loss_delay_penalty[
            "recent_unrealized_pnl_yen"
        ],
        loss_delay_no_recent_improvement=bool(
            loss_delay_penalty["no_recent_improvement"]
        ),
        loss_delay_penalty_cap_yen=float(loss_delay_penalty["cap_yen"]),
        loss_delay_penalty_cap_remaining_yen=loss_delay_penalty[
            "cap_remaining_yen"
        ],
        new_entry_penalty_applied=bool(new_entry_penalty["applied"]),
        new_entry_penalty_raw_yen=float(new_entry_penalty["penalty_raw_yen"]),
        new_entry_penalty_apply_to=str(new_entry_penalty["apply_to"]),
        new_entry_penalty_entry_side=new_entry_penalty["entry_side"],
    )


def _compute_loss_delay_penalty(
    *,
    position_before: Position,
    action: Action,
    order_book: dict,
    equity_after: Optional[float],
    is_forced_exit: bool,
    penalty_raw_yen: float,
    unrealized_loss_threshold_yen: float,
    require_close_executable: bool,
    entry_cash_pnl_before: Optional[float],
    loss_state_steps_before: int,
    min_persistence_steps: int,
    holding_time_seconds: float,
    min_holding_time_seconds: float,
    require_no_recent_loss_improvement: bool,
    recent_unrealized_pnl_yen: Optional[float],
    roundtrip_penalty_accrued_yen: float,
    per_roundtrip_cap_yen: float,
) -> dict[str, object]:
    penalty = _finite_nonnegative_float(penalty_raw_yen)
    threshold = _finite_nonnegative_float(unrealized_loss_threshold_yen)
    min_persistence = max(1, _nonnegative_int(min_persistence_steps))
    holding_seconds = _finite_nonnegative_float_or_zero(holding_time_seconds)
    min_holding_seconds = _finite_nonnegative_float_or_zero(
        min_holding_time_seconds
    )
    accrued = _finite_nonnegative_float_or_zero(roundtrip_penalty_accrued_yen)
    cap = _finite_nonnegative_float_or_zero(per_roundtrip_cap_yen)
    close_action: Action | None = None
    if position_before == "Long":
        close_action = "Sell"
        close_book_side: BookSide = "BuyTop5"
    elif position_before == "Short":
        close_action = "Buy"
        close_book_side = "SellTop5"
    else:
        close_book_side = "none"

    close_executable = False
    if close_action is not None:
        close_execution = compute_top5_weighted_price(
            _extract_top5_levels(order_book, close_book_side),
            qty=ORDER_QTY,
            used_book_side=close_book_side,
        )
        close_executable = bool(close_execution.executable)

    unrealized = _current_roundtrip_unrealized_pnl(
        equity_after=equity_after,
        entry_cash_pnl_before=entry_cash_pnl_before,
    )
    in_loss_state = unrealized is not None and threshold > 0.0 and unrealized <= -threshold
    loss_state_steps = (
        max(0, _nonnegative_int(loss_state_steps_before)) + 1
        if in_loss_state
        else 0
    )
    recent_unrealized = _finite_float_or_none(recent_unrealized_pnl_yen)
    if bool(require_no_recent_loss_improvement):
        no_recent_improvement = (
            recent_unrealized is not None
            and unrealized is not None
            and unrealized <= recent_unrealized
        )
    else:
        no_recent_improvement = True
    cap_remaining = None
    capped_penalty = penalty
    if cap > 0.0:
        cap_remaining = max(0.0, cap - accrued)
        capped_penalty = min(penalty, cap_remaining)
    applied = (
        capped_penalty > 0.0
        and threshold > 0.0
        and not bool(is_forced_exit)
        and action == "Hold"
        and position_before in {"Long", "Short"}
        and in_loss_state
        and loss_state_steps >= min_persistence
        and holding_seconds >= min_holding_seconds
        and no_recent_improvement
        and (close_executable or not bool(require_close_executable))
    )
    return {
        "applied": bool(applied),
        "penalty_raw_yen": capped_penalty if applied else 0.0,
        "threshold_yen": threshold if threshold > 0.0 else None,
        "unrealized_pnl_yen": unrealized,
        "close_action": close_action,
        "close_executable": close_executable,
        "loss_state_steps": loss_state_steps,
        "min_persistence_steps": min_persistence,
        "holding_time_seconds": holding_seconds,
        "min_holding_time_seconds": min_holding_seconds,
        "recent_unrealized_pnl_yen": recent_unrealized,
        "no_recent_improvement": no_recent_improvement,
        "cap_yen": cap,
        "cap_remaining_yen": cap_remaining,
    }


def _compute_new_entry_penalty(
    *,
    position_before: Position,
    position_after: Position,
    action: Action,
    executed: bool,
    execution_failed: bool,
    is_forced_exit: bool,
    penalty_raw_yen: float,
    apply_to: str,
) -> dict[str, object]:
    penalty = _finite_nonnegative_float(penalty_raw_yen)
    target = _normalize_new_entry_penalty_apply_to(apply_to)
    entry_side: str | None = None
    if position_before == "Flat" and action == "Buy" and position_after == "Long":
        entry_side = "Long"
    elif position_before == "Flat" and action == "Sell" and position_after == "Short":
        entry_side = "Short"
    applies_to_side = (
        target == "both"
        or (target == "long_only" and entry_side == "Long")
        or (target == "short_only" and entry_side == "Short")
    )
    applied = (
        penalty > 0.0
        and target != "off"
        and entry_side is not None
        and applies_to_side
        and bool(executed)
        and not bool(execution_failed)
        and not bool(is_forced_exit)
    )
    return {
        "applied": bool(applied),
        "penalty_raw_yen": penalty if applied else 0.0,
        "apply_to": target,
        "entry_side": entry_side,
    }


def _normalize_new_entry_penalty_apply_to(value: str) -> str:
    normalized = str(value)
    if normalized in {"off", "both", "long_only", "short_only"}:
        return normalized
    return "off"


def _finite_nonnegative_float(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric) or numeric <= 0.0:
        return 0.0
    return numeric


def _finite_nonnegative_float_or_zero(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric) or numeric < 0.0:
        return 0.0
    return numeric


def _finite_float_or_none(value: Optional[float]) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _nonnegative_int(value: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, numeric)


def _current_roundtrip_unrealized_pnl(
    *,
    equity_after: Optional[float],
    entry_cash_pnl_before: Optional[float],
) -> Optional[float]:
    equity = _finite_float_or_none(equity_after)
    if equity is None:
        return None
    entry_cash = _finite_float_or_none(entry_cash_pnl_before)
    if entry_cash is None:
        return equity
    return equity - entry_cash


def _extract_top5_levels(order_book: dict, used_book_side: BookSide) -> list[OrderBookLevel]:
    if used_book_side == "SellTop5":
        prefix = "Sell"
    elif used_book_side == "BuyTop5":
        prefix = "Buy"
    else:
        return []

    return [
        OrderBookLevel(
            price=order_book.get(f"{prefix}{level}_Price"),
            qty=order_book.get(f"{prefix}{level}_Qty"),
        )
        for level in range(1, 6)
    ]


def _join_failure_reasons(*reasons: Optional[str]) -> Optional[str]:
    non_empty_reasons = [reason for reason in reasons if reason]
    return ";".join(non_empty_reasons) if non_empty_reasons else None
