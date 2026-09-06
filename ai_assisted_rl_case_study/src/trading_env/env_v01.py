"""Minimal TradingEnv v0.1.

This environment consumes one precomputed episode: one symbol for one trading
day. It does not load CSVs, generate features, or infer trading-session rules.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
import math
from typing import Any

from .execution_latency_v01 import (
    ENVIRONMENT_CONTRACT_ID,
    EXECUTION_CONTRACT_ID,
    REWARD_CLOCK_CONTRACT_ID,
    SIDECAR_SCHEMA_ID,
    TRANSITION_CONTRACT_ID,
    TRADING_ENV_KWARGS_ATTR,
    normalize_timestamp_jst_v01,
    validate_execution_payload_v01,
)
from .execution_closingrisk_v02 import (
    ENVIRONMENT_CONTRACT_ID as CLOSINGRISK_ENVIRONMENT_CONTRACT_ID,
    EXECUTION_CONTRACT_ID as CLOSINGRISK_EXECUTION_CONTRACT_ID,
    REWARD_CLOCK_CONTRACT_ID as CLOSINGRISK_REWARD_CLOCK_CONTRACT_ID,
    SIDECAR_SCHEMA_ID as CLOSINGRISK_SIDECAR_SCHEMA_ID,
    TRANSITION_CONTRACT_ID as CLOSINGRISK_TRANSITION_CONTRACT_ID,
    execution_info_v02,
    policy_effective_action_mask_v02,
    resolve_forced_liquidation_v02,
    risk_management_action_v02,
    validate_execution_closingrisk_sidecar_v02,
    validate_execution_payload_v02,
)
from .reward_v01 import (
    NAN,
    StepRewardResult,
    step_reward_v01,
    validate_reference_notional,
)
from .state_builder_v01 import build_env_state_vector_v01
from features.state_vector_v01 import (
    get_agent_transition_history_depth_v01,
    get_state_candidate_name_v01,
)


class TradingEnv:
    """A minimal single-symbol, single-day trading environment."""

    def __init__(
        self,
        episode_rows,
        step_seconds: int = 5,
        *,
        entry_reentry_cooldown_steps: int = 0,
        entry_reentry_cooldown_apply_to: str = "off",
        entry_reentry_cooldown_state_enabled: bool = False,
        entry_reentry_cooldown_session_boundary: str = "continue",
        entry_reentry_cooldown_trigger_forced_exit: bool = True,
        loss_delay_penalty_raw_yen: float = 0.0,
        loss_delay_unrealized_loss_threshold_yen: float = 50_000.0,
        loss_delay_require_close_executable: bool = True,
        loss_delay_min_persistence_steps: int = 1,
        loss_delay_min_holding_time_seconds: float = 0.0,
        loss_delay_require_no_recent_loss_improvement: bool = False,
        loss_delay_recent_loss_improvement_lookback_steps: int = 0,
        loss_delay_per_roundtrip_cap_yen: float = 0.0,
        new_entry_penalty_raw_yen: float = 0.0,
        new_entry_penalty_apply_to: str = "off",
        state_candidate_name: str | None = None,
        execution_contract_id: str | None = None,
        execution_sidecar: Any | None = None,
        _historical_zero_latency_v01: bool = False,
    ):
        attached_runtime = dict(getattr(episode_rows, "attrs", {})).get(
            TRADING_ENV_KWARGS_ATTR
        )
        if execution_contract_id is None and execution_sidecar is None:
            if isinstance(attached_runtime, dict):
                execution_contract_id = attached_runtime.get(
                    "execution_contract_id"
                )
                execution_sidecar = attached_runtime.get("execution_sidecar")
        if hasattr(episode_rows, "to_dict"):
            self.episode_rows = episode_rows.to_dict("records")
        else:
            self.episode_rows = list(episode_rows)
        if not self.episode_rows:
            raise ValueError("episode_rows must not be empty")
        if not isinstance(step_seconds, int) or step_seconds <= 0:
            raise ValueError("step_seconds must be a positive integer")
        self.execution_contract_id = _validate_execution_sidecar_opt_in(
            execution_contract_id=execution_contract_id,
            execution_sidecar=execution_sidecar,
            episode_rows=episode_rows,
            episode_row_count=len(self.episode_rows),
            historical_zero_latency=bool(_historical_zero_latency_v01),
        )
        self.execution_sidecar = execution_sidecar
        self.entry_reentry_cooldown_steps = _validate_cooldown_steps(
            entry_reentry_cooldown_steps
        )
        self.entry_reentry_cooldown_apply_to = _validate_cooldown_apply_to(
            entry_reentry_cooldown_apply_to
        )
        self.entry_reentry_cooldown_state_enabled = bool(
            entry_reentry_cooldown_state_enabled
        )
        self.entry_reentry_cooldown_session_boundary = (
            _validate_cooldown_session_boundary(
                entry_reentry_cooldown_session_boundary
            )
        )
        self.entry_reentry_cooldown_trigger_forced_exit = bool(
            entry_reentry_cooldown_trigger_forced_exit
        )
        self.loss_delay_penalty_raw_yen = _validate_nonnegative_float(
            loss_delay_penalty_raw_yen,
            "loss_delay_penalty_raw_yen",
        )
        self.loss_delay_unrealized_loss_threshold_yen = _validate_nonnegative_float(
            loss_delay_unrealized_loss_threshold_yen,
            "loss_delay_unrealized_loss_threshold_yen",
        )
        self.loss_delay_require_close_executable = bool(
            loss_delay_require_close_executable
        )
        self.loss_delay_min_persistence_steps = _validate_nonnegative_int(
            loss_delay_min_persistence_steps,
            "loss_delay_min_persistence_steps",
        )
        if self.loss_delay_min_persistence_steps <= 0:
            self.loss_delay_min_persistence_steps = 1
        self.loss_delay_min_holding_time_seconds = _validate_nonnegative_float(
            loss_delay_min_holding_time_seconds,
            "loss_delay_min_holding_time_seconds",
        )
        self.loss_delay_require_no_recent_loss_improvement = bool(
            loss_delay_require_no_recent_loss_improvement
        )
        self.loss_delay_recent_loss_improvement_lookback_steps = (
            _validate_nonnegative_int(
                loss_delay_recent_loss_improvement_lookback_steps,
                "loss_delay_recent_loss_improvement_lookback_steps",
            )
        )
        self.loss_delay_per_roundtrip_cap_yen = _validate_nonnegative_float(
            loss_delay_per_roundtrip_cap_yen,
            "loss_delay_per_roundtrip_cap_yen",
        )
        self.new_entry_penalty_raw_yen = _validate_nonnegative_float(
            new_entry_penalty_raw_yen,
            "new_entry_penalty_raw_yen",
        )
        self.new_entry_penalty_apply_to = _validate_new_entry_penalty_apply_to(
            new_entry_penalty_apply_to
        )
        self.state_candidate_name = get_state_candidate_name_v01(
            state_candidate_name
        )
        self.agent_transition_history_depth = get_agent_transition_history_depth_v01(
            self.state_candidate_name
        )
        self.agent_transition_history_enabled = self.agent_transition_history_depth > 0
        if self.entry_reentry_cooldown_steps > 0 and (
            self.entry_reentry_cooldown_apply_to == "off"
        ):
            raise ValueError(
                "entry_reentry_cooldown_apply_to must not be off when "
                "entry_reentry_cooldown_steps > 0"
            )

        self.step_seconds = step_seconds
        self.position = "Flat"
        self.cash_pnl = 0.0
        self.last_valid_equity = 0.0
        self.entry_price = None
        self.entry_cash_pnl_before = None
        self.holding_time_seconds = 0.0
        self.loss_delay_loss_state_steps = 0
        self.loss_delay_roundtrip_penalty_accrued_yen = 0.0
        self.loss_delay_unrealized_history = deque()
        self.cooldown_remaining_steps = 0
        self.index = 0
        self.done = False
        self._last_info_context: dict[str, Any] = {}
        self._last_state_result = None
        self._previous_decision_position: str | None = None
        self._previous_effective_action: str | None = None
        self._agent_transition_history: list[dict[str, str]] = []

    @classmethod
    def for_historical_zero_latency_v01(cls, episode_rows, **kwargs):
        """Explicit Git/history reproduction route; never a current default."""

        kwargs["_historical_zero_latency_v01"] = True
        return cls(episode_rows, **kwargs)

    def reset(self):
        self.index = 0
        self.position = "Flat"
        self.cash_pnl = 0.0
        self.last_valid_equity = 0.0
        self.entry_price = None
        self.entry_cash_pnl_before = None
        self.holding_time_seconds = 0.0
        self.loss_delay_loss_state_steps = 0
        self.loss_delay_roundtrip_penalty_accrued_yen = 0.0
        self.loss_delay_unrealized_history.clear()
        self.cooldown_remaining_steps = 0
        self.done = False
        self._last_info_context = {
            "cooldown_reset_count": 1,
            "cooldown_reset_reason": "episode_reset",
        }
        self._last_state_result = None
        self._previous_decision_position = None
        self._previous_effective_action = None
        self._agent_transition_history = []
        return self._get_state(0)

    def step(self, action: str):
        if self.done:
            raise RuntimeError("step() called after episode is done")
        if self.agent_transition_history_enabled and action not in {
            "Hold",
            "Buy",
            "Sell",
        }:
            raise ValueError(
                "Agent History effective action must be semantic Hold, Buy, or Sell"
            )

        row = self.episode_rows[self.index]
        current_index = self.index
        agent_history_before = self.get_agent_transition_history_context()
        position_before = self.position
        cash_pnl_before = self.cash_pnl
        equity_before = self.last_valid_equity
        holding_time_seconds_before = float(self.holding_time_seconds)
        cooldown_remaining_before = int(self.cooldown_remaining_steps)
        cooldown_active_before = cooldown_remaining_before > 0

        previous_close = row.get("previous_close", row.get("PreviousClose"))
        is_forced_exit = bool(row.get("is_forced_exit", False))
        loss_delay_recent_unrealized = self._loss_delay_recent_unrealized_pnl()
        loss_delay_holding_time_seconds = self._loss_delay_holding_time_for_action(
            action
        )

        execution_payload = None
        execution_routing_failure = None
        if self.execution_contract_id is None:
            order_book = self._build_order_book(row)
        else:
            try:
                execution_payload = self.execution_sidecar.payload_at(current_index)
            except (IndexError, KeyError, TypeError, ValueError):
                execution_payload = None
            if self.execution_contract_id == CLOSINGRISK_EXECUTION_CONTRACT_ID:
                execution_routing_failure = validate_execution_payload_v02(
                    payload=execution_payload,
                    decision_row=row,
                )
            else:
                execution_routing_failure = validate_execution_payload_v01(
                    payload=execution_payload,
                    decision_row=row,
                )
            order_book = (
                self._build_order_book(execution_payload)
                if execution_routing_failure is None
                else None
            )

        closing_diagnostics = None
        if execution_routing_failure is not None:
            result = _execution_routing_failure_result_v01(
                position_before=self.position,
                cash_pnl_before=self.cash_pnl,
                action=action,
                previous_close=previous_close,
                is_forced_exit=is_forced_exit,
                reason=execution_routing_failure,
            )
        elif (
            self.execution_contract_id == CLOSINGRISK_EXECUTION_CONTRACT_ID
            and is_forced_exit
        ):
            assert execution_payload is not None
            result, closing_diagnostics = resolve_forced_liquidation_v02(
                position_before=self.position,
                cash_pnl_before=self.cash_pnl,
                equity_before=self.last_valid_equity,
                previous_close=previous_close,
                payload=execution_payload,
            )
        else:
            assert order_book is not None
            result = step_reward_v01(
                position_before=self.position,
                cash_pnl_before=self.cash_pnl,
                equity_before=self.last_valid_equity,
                action=action,
                order_book=order_book,
                previous_close=previous_close,
                is_forced_exit=is_forced_exit,
                loss_delay_penalty_raw_yen=self.loss_delay_penalty_raw_yen,
                loss_delay_unrealized_loss_threshold_yen=(
                    self.loss_delay_unrealized_loss_threshold_yen
                ),
                loss_delay_require_close_executable=(
                    self.loss_delay_require_close_executable
                ),
                loss_delay_entry_cash_pnl_before=self.entry_cash_pnl_before,
                loss_delay_loss_state_steps_before=(
                    self.loss_delay_loss_state_steps
                ),
                loss_delay_min_persistence_steps=(
                    self.loss_delay_min_persistence_steps
                ),
                loss_delay_holding_time_seconds=loss_delay_holding_time_seconds,
                loss_delay_min_holding_time_seconds=(
                    self.loss_delay_min_holding_time_seconds
                ),
                loss_delay_require_no_recent_loss_improvement=(
                    self.loss_delay_require_no_recent_loss_improvement
                ),
                loss_delay_recent_unrealized_pnl_yen=loss_delay_recent_unrealized,
                loss_delay_roundtrip_penalty_accrued_yen=(
                    self.loss_delay_roundtrip_penalty_accrued_yen
                ),
                loss_delay_per_roundtrip_cap_yen=(
                    self.loss_delay_per_roundtrip_cap_yen
                ),
                new_entry_penalty_raw_yen=self.new_entry_penalty_raw_yen,
                new_entry_penalty_apply_to=self.new_entry_penalty_apply_to,
            )

        self.position = result.position_after
        self.cash_pnl = result.cash_pnl_after
        if result.reward_valid and result.equity_after is not None:
            self.last_valid_equity = result.equity_after
        self._update_position_tracking(
            position_before=position_before,
            cash_pnl_before=cash_pnl_before,
            result=result,
        )
        self._update_loss_delay_tracking(result)
        cooldown_update = self._update_cooldown_after_step(
            position_before=position_before,
            result=result,
            cooldown_remaining_before=cooldown_remaining_before,
        )

        self.index += 1
        self.done = self.index >= len(self.episode_rows)
        agent_history_next = {
            "enabled": self.agent_transition_history_enabled,
            "depth": self.agent_transition_history_depth,
            "available": False,
            "previous_decision_position": None,
            "previous_effective_action": None,
            "groups": [
                {
                    "lag": lag,
                    "available": False,
                    "position_before": None,
                    "effective_action": None,
                }
                for lag in range(1, self.agent_transition_history_depth + 1)
            ],
        }
        if not self.done:
            next_row = self.episode_rows[self.index]
            same_session = (
                row.get("session") is not None
                and row.get("session") == next_row.get("session")
            )
            if (
                self.agent_transition_history_enabled
                and not is_forced_exit
                and same_session
            ):
                self._agent_transition_history.insert(
                    0,
                    {
                        "position_before": position_before,
                        "effective_action": action,
                    },
                )
                del self._agent_transition_history[
                    self.agent_transition_history_depth:
                ]
            else:
                self._agent_transition_history = []
            if self._agent_transition_history:
                self._previous_decision_position = self._agent_transition_history[0][
                    "position_before"
                ]
                self._previous_effective_action = self._agent_transition_history[0][
                    "effective_action"
                ]
            else:
                self._previous_decision_position = None
                self._previous_effective_action = None
            agent_history_next = self.get_agent_transition_history_context()
        if self.done:
            next_state = None
            state_result_for_info = self._build_state_result(current_index)
        else:
            state_result_for_info = self._build_state_result(self.index)
            next_state = state_result_for_info.state_vector
            self._last_state_result = state_result_for_info

        self._last_info_context = {
            "index": current_index,
            "symbol": row.get("symbol"),
            "timestamp": row.get("timestamp", row.get("grid_timestamp")),
            "session": row.get("session"),
            "position_before": position_before,
            "cash_pnl_before": cash_pnl_before,
            "equity_before": equity_before,
            "holding_time_seconds_before": holding_time_seconds_before,
            "position_after": result.position_after,
            "cash_pnl_after": result.cash_pnl_after,
            "equity_after": result.equity_after,
            "done": self.done,
            "cooldown_remaining_before": cooldown_remaining_before,
            "cooldown_remaining_after": int(self.cooldown_remaining_steps),
            "cooldown_active_before": cooldown_active_before,
            "cooldown_active_after": self.cooldown_active,
            "cooldown_triggered": bool(cooldown_update["triggered"]),
            "cooldown_trigger_reason": cooldown_update["trigger_reason"],
            "cooldown_triggered_count": int(cooldown_update["triggered"]),
            "cooldown_triggered_by_long_close_count": int(
                cooldown_update["trigger_reason"] == "long_close"
            ),
            "cooldown_triggered_by_short_close_count": int(
                cooldown_update["trigger_reason"] == "short_close"
            ),
            "cooldown_triggered_by_forced_exit_count": int(
                cooldown_update["trigger_reason"] == "forced_exit"
            ),
            "cooldown_active_step_count": int(cooldown_active_before),
            "cooldown_reset_count": 0,
            "cooldown_reset_reason": None,
            "state_dim": state_result_for_info.qc["state_dim"],
            "state_columns": state_result_for_info.state_columns,
            "missing_count": state_result_for_info.qc["missing_count"],
            "nonfinite_count": state_result_for_info.qc["nonfinite_count"],
            "valid_state_mask": state_result_for_info.qc["valid_state_mask"],
            "state_schema_id": state_result_for_info.qc["state_schema_id"],
            "state_candidate_name": self.state_candidate_name,
            "agent_transition_history_before": agent_history_before,
            "agent_transition_history_next": agent_history_next,
        }
        if self.execution_contract_id is not None:
            if self.execution_contract_id == CLOSINGRISK_EXECUTION_CONTRACT_ID:
                self._last_info_context.update(
                    execution_info_v02(
                        payload=execution_payload,
                        result=result,
                        routing_failure=execution_routing_failure,
                        closing_diagnostics=closing_diagnostics,
                    )
                )
            else:
                self._last_info_context.update(
                    _execution_latency_info_v01(
                        payload=execution_payload,
                        result=result,
                        routing_failure=execution_routing_failure,
                    )
                )
        info = self._build_info(result)

        return next_state, result.reward, self.done, info

    def _build_order_book(self, row: dict) -> dict:
        order_book = {}
        for side in ("Sell", "Buy"):
            for level in range(1, 6):
                order_book[f"{side}{level}_Price"] = row.get(
                    f"{side}{level}_Price"
                )
                order_book[f"{side}{level}_Qty"] = row.get(f"{side}{level}_Qty")
        return order_book

    def _get_state(self, index: int):
        state_result = self._build_state_result(index)
        self._last_state_result = state_result
        return state_result.state_vector

    def _build_state_result(self, index: int):
        return build_env_state_vector_v01(
            row=self.episode_rows[index],
            position_side=self.position,
            entry_price=self.entry_price,
            holding_time_seconds=self.holding_time_seconds,
            cooldown_remaining_steps=self.cooldown_remaining_steps,
            cooldown_length_steps=self.entry_reentry_cooldown_steps,
            state_candidate_name=self.state_candidate_name,
            previous_decision_position=self._previous_decision_position,
            previous_effective_action=self._previous_effective_action,
            agent_transition_history=tuple(
                dict(item) for item in self._agent_transition_history
            ),
        )

    def get_agent_transition_history_context(self) -> dict[str, Any]:
        """Return the candidate history currently visible to the policy state."""

        available = bool(self.agent_transition_history_enabled and self._agent_transition_history)
        groups = []
        for index in range(self.agent_transition_history_depth):
            item = (
                self._agent_transition_history[index]
                if index < len(self._agent_transition_history)
                else None
            )
            groups.append(
                {
                    "lag": index + 1,
                    "available": item is not None,
                    "position_before": (
                        item["position_before"] if item is not None else None
                    ),
                    "effective_action": (
                        item["effective_action"] if item is not None else None
                    ),
                }
            )
        return {
            "enabled": self.agent_transition_history_enabled,
            "depth": self.agent_transition_history_depth,
            "available": available,
            "previous_decision_position": (
                self._previous_decision_position if available else None
            ),
            "previous_effective_action": (
                self._previous_effective_action if available else None
            ),
            "groups": groups,
        }

    def _update_position_tracking(
        self,
        *,
        position_before: str,
        cash_pnl_before: float,
        result,
    ) -> None:
        opened_position = (
            position_before == "Flat"
            and result.position_after in ("Long", "Short")
            and result.executed
            and result.execution_price is not None
        )
        if result.position_after == "Flat":
            self.entry_price = None
            self.entry_cash_pnl_before = None
            self.holding_time_seconds = 0.0
            return
        if opened_position:
            self.entry_price = result.execution_price
            self.entry_cash_pnl_before = cash_pnl_before
            self.holding_time_seconds = 0.0
            return
        if result.position_after == position_before:
            self.holding_time_seconds += float(self.step_seconds)

    def _loss_delay_holding_time_for_action(self, action: str) -> float:
        if self.position in {"Long", "Short"} and action == "Hold":
            return float(self.holding_time_seconds) + float(self.step_seconds)
        return float(self.holding_time_seconds)

    def _loss_delay_recent_unrealized_pnl(self) -> float | None:
        lookback = int(self.loss_delay_recent_loss_improvement_lookback_steps)
        if lookback <= 0 or len(self.loss_delay_unrealized_history) < lookback:
            return None
        return self.loss_delay_unrealized_history[-lookback]

    def _update_loss_delay_tracking(self, result) -> None:
        if result.loss_delay_penalty_applied:
            self.loss_delay_roundtrip_penalty_accrued_yen += float(
                result.loss_delay_penalty_raw_yen
            )
        if result.position_after not in {"Long", "Short"}:
            self.loss_delay_loss_state_steps = 0
            self.loss_delay_roundtrip_penalty_accrued_yen = 0.0
            self.loss_delay_unrealized_history.clear()
            return
        unrealized = self._current_roundtrip_unrealized_pnl(result)
        if unrealized is None:
            self.loss_delay_loss_state_steps = 0
            return
        if unrealized <= -float(self.loss_delay_unrealized_loss_threshold_yen):
            self.loss_delay_loss_state_steps += 1
        else:
            self.loss_delay_loss_state_steps = 0
        self.loss_delay_unrealized_history.append(float(unrealized))

    def _current_roundtrip_unrealized_pnl(self, result) -> float | None:
        if self.entry_cash_pnl_before is None or result.equity_after is None:
            return None
        try:
            equity = float(result.equity_after)
            entry_cash = float(self.entry_cash_pnl_before)
        except (TypeError, ValueError):
            return None
        if equity != equity or entry_cash != entry_cash:
            return None
        if equity in (float("inf"), float("-inf")):
            return None
        if entry_cash in (float("inf"), float("-inf")):
            return None
        return equity - entry_cash

    def get_action_mask(self) -> dict[str, bool]:
        """Return the position-owned environment mask (not the policy mask)."""

        masks = {
            "Flat": {"Buy": True, "Sell": True, "Hold": True},
            "Long": {"Buy": False, "Sell": True, "Hold": True},
            "Short": {"Buy": True, "Sell": False, "Hold": True},
        }
        return dict(masks[self.position])

    def get_policy_effective_action_mask(self) -> dict[str, bool]:
        """Return the current cutoff/risk-composed policy-effective mask."""

        env_mask = self.get_action_mask()
        q_order_mask = [
            bool(env_mask["Hold"]),
            bool(env_mask["Buy"]),
            bool(env_mask["Sell"]),
        ]
        row = self.episode_rows[self.index]
        result = policy_effective_action_mask_v02(
            env_action_mask=q_order_mask,
            position=self.position,
            decision_timestamp=row.get("grid_timestamp", row.get("timestamp")),
            is_forced_liquidation_event=bool(
                row.get("is_forced_exit", False)
            ),
        )
        effective = result["policy_effective_action_mask"]
        return {
            "Buy": bool(effective[1]),
            "Sell": bool(effective[2]),
            "Hold": bool(effective[0]),
        }

    def is_risk_management_event(self) -> bool:
        row = self.episode_rows[self.index]
        return bool(row.get("is_forced_exit", False))

    def get_risk_management_action(self) -> str:
        if not self.is_risk_management_event():
            raise RuntimeError("current row is not a risk-management event")
        return risk_management_action_v02(self.position)

    @property
    def cooldown_active(self) -> bool:
        return int(self.cooldown_remaining_steps) > 0

    def _update_cooldown_after_step(
        self,
        *,
        position_before: str,
        result,
        cooldown_remaining_before: int,
    ) -> dict[str, Any]:
        trigger_reason = self._cooldown_trigger_reason(position_before, result)
        if trigger_reason is not None:
            self.cooldown_remaining_steps = int(self.entry_reentry_cooldown_steps)
            return {"triggered": True, "trigger_reason": trigger_reason}
        self.cooldown_remaining_steps = max(0, int(cooldown_remaining_before) - 1)
        return {"triggered": False, "trigger_reason": None}

    def _cooldown_trigger_reason(self, position_before: str, result) -> str | None:
        if (
            self.entry_reentry_cooldown_steps <= 0
            or self.entry_reentry_cooldown_apply_to == "off"
        ):
            return None
        if self.entry_reentry_cooldown_apply_to != "any_direction_after_close":
            return None
        if not bool(result.executed) or bool(result.execution_failed):
            return None
        if str(result.position_after) != "Flat":
            return None
        if bool(result.is_forced_exit):
            if self.entry_reentry_cooldown_trigger_forced_exit:
                return "forced_exit"
            return None
        if position_before == "Long":
            return "long_close"
        if position_before == "Short":
            return "short_close"
        return None

    def _build_info(self, result) -> dict:
        info = asdict(result)
        info.update(self._last_info_context)
        return info


def _validate_cooldown_steps(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError("entry_reentry_cooldown_steps must be a nonnegative integer")
    try:
        steps = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "entry_reentry_cooldown_steps must be a nonnegative integer"
        ) from exc
    if steps < 0:
        raise ValueError("entry_reentry_cooldown_steps must be a nonnegative integer")
    return steps


def _validate_cooldown_apply_to(value: str) -> str:
    normalized = str(value)
    if normalized not in {"off", "any_direction_after_close"}:
        raise ValueError(
            "entry_reentry_cooldown_apply_to must be one of "
            "['any_direction_after_close', 'off']"
        )
    return normalized


def _validate_cooldown_session_boundary(value: str) -> str:
    normalized = str(value)
    if normalized not in {"continue", "reset"}:
        raise ValueError(
            "entry_reentry_cooldown_session_boundary must be one of "
            "['continue', 'reset']"
        )
    return normalized


def _validate_nonnegative_float(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative finite float")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative finite float") from exc
    if numeric < 0.0:
        raise ValueError(f"{name} must be a nonnegative finite float")
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be a nonnegative finite float")
    return numeric


def _validate_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if numeric < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return numeric


def _validate_new_entry_penalty_apply_to(value: str) -> str:
    normalized = str(value)
    if normalized not in {"off", "both", "long_only", "short_only"}:
        raise ValueError(
            "new_entry_penalty_apply_to must be one of "
            "['both', 'long_only', 'off', 'short_only']"
        )
    return normalized


def _validate_execution_sidecar_opt_in(
    *,
    execution_contract_id: str | None,
    execution_sidecar: Any | None,
    episode_rows: Any,
    episode_row_count: int,
    historical_zero_latency: bool,
) -> str | None:
    if execution_contract_id is None and execution_sidecar is None:
        if historical_zero_latency:
            return None
        raise ValueError(
            "current TradingEnv requires the latency1s closing-risk sidecar; "
            "use for_historical_zero_latency_v01 only for explicit history reproduction"
        )
    supported = {
        EXECUTION_CONTRACT_ID: (
            SIDECAR_SCHEMA_ID,
            ENVIRONMENT_CONTRACT_ID,
            REWARD_CLOCK_CONTRACT_ID,
            TRANSITION_CONTRACT_ID,
        ),
        CLOSINGRISK_EXECUTION_CONTRACT_ID: (
            CLOSINGRISK_SIDECAR_SCHEMA_ID,
            CLOSINGRISK_ENVIRONMENT_CONTRACT_ID,
            CLOSINGRISK_REWARD_CLOCK_CONTRACT_ID,
            CLOSINGRISK_TRANSITION_CONTRACT_ID,
        ),
    }
    if execution_contract_id not in supported:
        raise ValueError("unsupported execution latency contract")
    if execution_sidecar is None:
        raise ValueError("execution sidecar is required for latency opt-in")
    schema, environment, reward_clock, transition = supported[
        str(execution_contract_id)
    ]
    if getattr(execution_sidecar, "schema_id", None) != schema:
        raise ValueError("execution sidecar schema mismatch")
    if getattr(execution_sidecar, "execution_contract_id", None) != execution_contract_id:
        raise ValueError("execution sidecar contract mismatch")
    if getattr(execution_sidecar, "environment_contract_id", None) != environment:
        raise ValueError("execution sidecar environment identity mismatch")
    if getattr(execution_sidecar, "reward_clock_contract_id", None) != reward_clock:
        raise ValueError("execution sidecar reward clock identity mismatch")
    if getattr(execution_sidecar, "transition_contract_id", None) != transition:
        raise ValueError("execution sidecar transition identity mismatch")
    if len(execution_sidecar) != int(episode_row_count):
        raise ValueError("execution sidecar row count mismatch")
    if execution_contract_id == CLOSINGRISK_EXECUTION_CONTRACT_ID:
        if not hasattr(episode_rows, "columns"):
            raise ValueError(
                "current closing-risk episode rows must be a pandas DataFrame"
            )
        validate_execution_closingrisk_sidecar_v02(
            episode_rows,
            execution_sidecar,
        )
    return str(execution_contract_id)


def _execution_routing_failure_result_v01(
    *,
    position_before: str,
    cash_pnl_before: float,
    action: str,
    previous_close: Any,
    is_forced_exit: bool,
    reason: str,
) -> StepRewardResult:
    forced_failure = bool(is_forced_exit and position_before in {"Long", "Short"})
    reference = validate_reference_notional(previous_close)
    attempted_side = _attempted_execution_side_v01(
        position_before=position_before,
        action=action,
        is_forced_exit=is_forced_exit,
    )
    valuation_side = (
        "BuyTop5"
        if position_before == "Long" and not forced_failure
        else "SellTop5"
        if position_before == "Short" and not forced_failure
        else "none"
    )
    return StepRewardResult(
        position_after=position_before,
        cash_pnl_after=float(cash_pnl_before),
        equity_after=None,
        reward_raw_yen=None,
        reward=NAN,
        reward_valid=False,
        training_valid=False,
        executed=False,
        execution_failed=True,
        execution_price=None,
        execution_qty=0,
        invalid_action=False,
        valuation_failed=bool(position_before in {"Long", "Short"} and not forced_failure),
        is_forced_exit=bool(is_forced_exit),
        forced_exit_failed=forced_failure,
        execution_used_book_side=attempted_side,
        valuation_used_book_side=valuation_side,
        failure_reason=(f"forced_exit_failed:{reason}" if forced_failure else reason),
        reference_notional=reference.reference_notional,
        review_required=True,
    )


def _attempted_execution_side_v01(
    *, position_before: str, action: str, is_forced_exit: bool
) -> str:
    if is_forced_exit and position_before == "Long":
        return "BuyTop5"
    if is_forced_exit and position_before == "Short":
        return "SellTop5"
    if (position_before, action) in {("Flat", "Buy"), ("Short", "Buy")}:
        return "SellTop5"
    if (position_before, action) in {("Flat", "Sell"), ("Long", "Sell")}:
        return "BuyTop5"
    return "none"


def _execution_latency_info_v01(
    *,
    payload: dict[str, Any] | None,
    result: StepRewardResult,
    routing_failure: str | None,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "execution_contract_id": EXECUTION_CONTRACT_ID,
        "environment_contract_id": ENVIRONMENT_CONTRACT_ID,
        "reward_clock_contract_id": REWARD_CLOCK_CONTRACT_ID,
        "transition_contract_id": TRANSITION_CONTRACT_ID,
        "execution_only_payload_schema_id": SIDECAR_SCHEMA_ID,
        "execution_routing_failed": routing_failure is not None,
        "execution_routing_failure_reason": routing_failure,
        "execution_snapshot_count": int(routing_failure is None),
        "latency_policy_selected_execution": bool(result.executed),
        "latency_policy_selected_action_side": None,
        "delayed_vs_decision_execution_price_delta_bps": None,
        "delayed_vs_decision_execution_price_delta_sign": None,
    }
    if not isinstance(payload, dict):
        info.update(
            {
                "execution_target_timestamp": None,
                "execution_source_timestamp": None,
                "execution_source_row_order": None,
                "execution_snapshot_age_seconds": None,
                "execution_snapshot_advanced": None,
                "execution_snapshot_same_as_decision": None,
                "latency_time_band": None,
            }
        )
        return info
    for source, target in (
        ("execution_target_timestamp", "execution_target_timestamp"),
        ("execution_source_timestamp", "execution_source_timestamp"),
    ):
        try:
            info[target] = normalize_timestamp_jst_v01(payload[source]).isoformat()
        except (KeyError, TypeError, ValueError):
            info[target] = None
    info.update(
        {
            "execution_source_row_order": _optional_int(
                payload.get("execution_source_row_order")
            ),
            "execution_snapshot_age_seconds": _optional_finite_float(
                payload.get("execution_snapshot_age_seconds")
            ),
            "execution_snapshot_advanced": _optional_bool(
                payload.get("execution_snapshot_advanced")
            ),
            "execution_snapshot_same_as_decision": _optional_bool(
                payload.get("execution_snapshot_same_as_decision")
            ),
            "latency_time_band": payload.get("latency_time_band"),
        }
    )
    if result.executed and result.execution_used_book_side in {"SellTop5", "BuyTop5"}:
        side = "buy" if result.execution_used_book_side == "SellTop5" else "sell"
        delta = _optional_finite_float(
            payload.get(f"latency_{side}_price_delta_bps")
        )
        info["latency_policy_selected_action_side"] = side.capitalize()
        info["delayed_vs_decision_execution_price_delta_bps"] = delta
        if delta is not None:
            info["delayed_vs_decision_execution_price_delta_sign"] = (
                "positive" if delta > 0 else "negative" if delta < 0 else "zero"
            )
    return info


def _optional_finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if type(value) is bool:
        return value
    if type(value).__name__ == "bool_":
        return bool(value)
    return None
