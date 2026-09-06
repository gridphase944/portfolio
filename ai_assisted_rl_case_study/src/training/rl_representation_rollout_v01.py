"""Common readiness action gate and bounded rollout accounting for E0."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import math

from trading_env.execution_closingrisk_v02 import (
    EXECUTION_CONTRACT_ID as CLOSINGRISK_EXECUTION_CONTRACT_ID,
)
from training.rl_representation_integration_v01 import (
    RepresentationReadinessResultV01,
    RepresentationVariantV01,
    contract_for_variant_v01,
)


WARMUP_CONTRACT_ID = "rl_representation_w300_warmup_hold_v01"
POLICY_STATUS_NO_READY = "rl_representation_no_policy_eligible_decision_v01"
VALID_ACTIONS = ("Hold", "Buy", "Sell")
VALID_POSITIONS = ("Flat", "Long", "Short")
NEXT_REPRESENTATION_UNREADY_EXCLUSION_REASON = (
    "next_representation_unready_nonterminal"
)
@dataclass(frozen=True)
class RepresentationActionDecisionV01:
    variant: str
    action: str
    source: str
    policy_eligible: bool
    readiness_schema_id: str
    readiness_reasons: tuple[str, ...]


def select_representation_action_v01(
    variant: RepresentationVariantV01 | str,
    readiness: RepresentationReadinessResultV01,
    policy_selector: Callable[[], str],
) -> RepresentationActionDecisionV01:
    """Return deterministic Hold before readiness without invoking policy/Q."""

    contract = contract_for_variant_v01(variant)
    if not isinstance(readiness, RepresentationReadinessResultV01):
        raise TypeError("readiness must be RepresentationReadinessResultV01")
    if not readiness.ready:
        return RepresentationActionDecisionV01(
            variant=contract.variant.value,
            action="Hold",
            source="warmup_deterministic_hold",
            policy_eligible=False,
            readiness_schema_id=readiness.schema_id,
            readiness_reasons=readiness.reasons,
        )
    action = str(policy_selector())
    if action not in VALID_ACTIONS:
        raise ValueError(f"policy selector returned invalid action: {action!r}")
    return RepresentationActionDecisionV01(
        variant=contract.variant.value,
        action=action,
        source="policy",
        policy_eligible=True,
        readiness_schema_id=readiness.schema_id,
        readiness_reasons=(),
    )


def exclude_policy_transition_for_next_readiness_v01(
    *,
    current_readiness: RepresentationReadinessResultV01,
    done: bool,
    next_readiness: RepresentationReadinessResultV01 | None,
) -> bool:
    """Apply the PM-approved symmetric Replay exclusion predicate."""

    if not isinstance(current_readiness, RepresentationReadinessResultV01):
        raise TypeError("current_readiness must be RepresentationReadinessResultV01")
    if type(done) is not bool:
        raise TypeError("done must be bool")
    if done:
        return False
    if not isinstance(next_readiness, RepresentationReadinessResultV01):
        raise TypeError("nonterminal next_readiness is required")
    return bool(current_readiness.ready and not next_readiness.ready)


class E0EpisodeAccountingV01:
    """Full-economy plus policy-only denominators from every Env step."""

    def __init__(self, *, episode_id: str, symbol: str, trading_date: str):
        if not episode_id or not symbol or not trading_date:
            raise ValueError("episode/symbol/date identities must not be empty")
        self.episode_id = str(episode_id)
        self.symbol = str(symbol)
        self.trading_date = str(trading_date)
        self.env_step_count = 0
        self.policy_decision_count = 0
        self.risk_management_step_count = 0
        self.warmup_step_count = 0
        self.full_normalized_reward = 0.0
        self.full_raw_yen_reward = 0.0
        self.warmup_normalized_reward = 0.0
        self.warmup_raw_yen_reward = 0.0
        self.policy_normalized_reward = 0.0
        self.policy_raw_yen_reward = 0.0
        self.policy_actions: Counter[str] = Counter()
        self.action_position: Counter[str] = Counter()
        self.position_occupancy: Counter[str] = Counter()
        self.entry_counts: Counter[str] = Counter()
        self.close_counts: Counter[str] = Counter()
        self.completed_holding_seconds: list[float] = []
        self.warmup_by_session: dict[str, Counter[str]] = defaultdict(Counter)
        self.session_aggregations: dict[str, Counter[str]] = defaultdict(Counter)
        self.first_policy_timestamp_by_session: dict[str, str] = {}
        self.execution_failure_reasons: Counter[str] = Counter()
        self.valuation_failure_reasons: Counter[str] = Counter()
        self.forced_exit_count = 0
        self.forced_exit_failure_count = 0
        self.invalid_action_count = 0
        self.review_required_count = 0
        self.training_valid = True
        self.final_cash_pnl: float | None = None
        self.final_equity: float | None = None
        self.reward_observation_valid_count = 0
        self.reward_observation_missing_count = 0
        self._open_entry_timestamp: datetime | None = None
        self._last_normal_close_timestamp: datetime | None = None
        self.timestamp_completed_holding_seconds: list[float] = []
        self.close_to_next_entry_seconds: list[float] = []
        self.normal_completed_trade_count = 0
        self.normal_close_within_5s_count = 0
        self.normal_close_count = 0
        self.normal_close_followed_by_reentry_within_5s_count = 0
        self.latency_contract_id: str | None = None
        self.latency_snapshot_count = 0
        self.latency_snapshot_advanced_count = 0
        self.latency_snapshot_same_count = 0
        self.latency_snapshot_ages: list[float] = []
        self.latency_forced_snapshot_ages: list[float] = []
        self.latency_routing_failure_reasons: Counter[str] = Counter()
        self.latency_actual_execution_count = 0
        self.latency_actual_execution_delta_missing_count = 0
        self.latency_actual_delta_by_side: dict[str, list[float]] = defaultdict(list)
        self.latency_actual_delta_sign_by_side: Counter[str] = Counter()
        self.latency_snapshot_by_session: Counter[str] = Counter()
        self.latency_snapshot_by_time_band: Counter[str] = Counter()
        self.entry_cutoff_active_step_count = 0
        self.entry_cutoff_blocked_action_slot_count = 0
        self.entry_cutoff_env_valid_action_slot_count = 0
        self.entry_cutoff_blocked_by_timestamp: Counter[str] = Counter()
        self.position_at_1515_before: str | None = None
        self.position_at_1515_after: str | None = None
        self.position_at_1520_before: str | None = None
        self.position_at_1520_after: str | None = None
        self.forced_liquidation_stages: Counter[str] = Counter()
        self.forced_liquidation_delays: list[float] = []
        self.closing_auction_use_count = 0
        self.forced_liquidation_unresolved_count = 0
        self.closingrisk_contract_active = False

    def observe_step(
        self,
        *,
        decision: RepresentationActionDecisionV01,
        timestamp: str,
        session: str,
        position_before: str,
        position_after: str,
        normalized_reward: float | None,
        raw_yen_reward: float | None,
        info: dict[str, Any],
    ) -> None:
        if decision.action not in VALID_ACTIONS:
            raise ValueError("invalid action")
        if position_before not in VALID_POSITIONS or position_after not in VALID_POSITIONS:
            raise ValueError("invalid position")
        normalized = _optional_finite_strict(normalized_reward, "normalized_reward")
        raw = _optional_finite_strict(raw_yen_reward, "raw_yen_reward")
        reward_observed = normalized is not None and raw is not None
        if (normalized is None) != (raw is None):
            raise ValueError("normalized/raw reward availability must match")
        self.reward_observation_valid_count += int(reward_observed)
        self.reward_observation_missing_count += int(not reward_observed)
        normalized_value = 0.0 if normalized is None else normalized
        raw_value = 0.0 if raw is None else raw
        self.env_step_count += 1
        self.full_normalized_reward += normalized_value
        self.full_raw_yen_reward += raw_value
        self.position_occupancy[position_after] += 1
        session_summary = self.session_aggregations[str(session)]
        session_summary["env_step_count"] += 1
        session_summary["full_normalized_reward"] += normalized_value
        session_summary["full_raw_yen_reward"] += raw_value
        session_summary["reward_observation_valid_count"] += int(reward_observed)
        session_summary["reward_observation_missing_count"] += int(not reward_observed)
        risk_management = decision.source == "risk_management_forced_liquidation"
        if decision.policy_eligible:
            self.policy_decision_count += 1
            self.policy_normalized_reward += normalized_value
            self.policy_raw_yen_reward += raw_value
            self.policy_actions[decision.action] += 1
            self.action_position[f"{decision.action}|{position_before}"] += 1
            self.first_policy_timestamp_by_session.setdefault(str(session), str(timestamp))
            session_summary["policy_eligible_decision_count"] += 1
            session_summary["policy_normalized_reward"] += normalized_value
            session_summary["policy_raw_yen_reward"] += raw_value
            session_summary[f"policy_{decision.action}_count"] += 1
        elif risk_management:
            self.risk_management_step_count += 1
            session_summary["risk_management_step_count"] += 1
            session_summary["risk_management_normalized_reward"] += normalized_value
            session_summary["risk_management_raw_yen_reward"] += raw_value
        else:
            if decision.action != "Hold" or decision.source != "warmup_deterministic_hold":
                raise ValueError("unready step must be deterministic warm-up Hold")
            self.warmup_step_count += 1
            self.warmup_normalized_reward += normalized_value
            self.warmup_raw_yen_reward += raw_value
            session_counts = self.warmup_by_session[str(session)]
            session_counts["step_count"] += 1
            session_counts["hold_count"] += 1
            session_counts["replay_excluded_count"] += 1
            session_counts[f"carried_{position_before}_step_count"] += int(
                position_before in {"Long", "Short"}
            )
            session_summary["warmup_step_count"] += 1
            session_summary["warmup_normalized_reward"] += normalized_value
            session_summary["warmup_raw_yen_reward"] += raw_value
        if position_before == "Flat" and position_after == "Long":
            self.entry_counts["Long"] += 1
            self._record_entry_timestamp(_parse_iso_timestamp(timestamp))
        elif position_before == "Flat" and position_after == "Short":
            self.entry_counts["Short"] += 1
            self._record_entry_timestamp(_parse_iso_timestamp(timestamp))
        elif position_before == "Long" and position_after == "Flat":
            self.close_counts["Long"] += 1
            self._record_holding(info)
            self._record_close_timestamp(
                self._effective_close_timestamp(timestamp, info), info
            )
        elif position_before == "Short" and position_after == "Flat":
            self.close_counts["Short"] += 1
            self._record_holding(info)
            self._record_close_timestamp(
                self._effective_close_timestamp(timestamp, info), info
            )
        failure_reason = str(info.get("failure_reason") or "unspecified")
        if bool(info.get("execution_failed", False)):
            self.execution_failure_reasons[failure_reason] += 1
        if bool(info.get("valuation_failed", False)):
            self.valuation_failure_reasons[failure_reason] += 1
        position_forced_exit = bool(info.get("is_forced_exit", False)) and (
            position_before in {"Long", "Short"}
        )
        self.forced_exit_count += int(position_forced_exit)
        self.forced_exit_failure_count += int(
            position_forced_exit and bool(info.get("forced_exit_failed", False))
        )
        self.invalid_action_count += int(bool(info.get("invalid_action", False)))
        self.review_required_count += int(bool(info.get("review_required", False)))
        self.training_valid = self.training_valid and bool(info.get("training_valid", True))
        self.final_cash_pnl = _optional_finite(info.get("cash_pnl_after"))
        self.final_equity = _optional_finite(info.get("equity_after"))
        if (
            info.get("execution_contract_id")
            == CLOSINGRISK_EXECUTION_CONTRACT_ID
        ):
            self.closingrisk_contract_active = True
            self._observe_closingrisk(
                timestamp=timestamp,
                position_before=position_before,
                position_after=position_after,
                info=info,
            )
        self._observe_latency(info=info, session=str(session))

    def finalize(self) -> dict[str, Any]:
        holding = _summary(self.completed_holding_seconds)
        action_counts = {name: int(self.policy_actions[name]) for name in VALID_ACTIONS}
        no_ready_sessions = sorted(
            set(self.session_aggregations) - set(self.first_policy_timestamp_by_session)
        )
        result = {
            "episode_id": self.episode_id,
            "symbol": self.symbol,
            "trading_date": self.trading_date,
            "policy_status": (
                "policy_eligible" if self.policy_decision_count else POLICY_STATUS_NO_READY
            ),
            "policy_status_reason": (
                None
                if self.policy_decision_count
                else "common_representation_readiness_never_true"
            ),
            "no_ready_session_count": len(no_ready_sessions),
            "no_ready_session_reasons": {
                session: "common_representation_readiness_never_true"
                for session in no_ready_sessions
            },
            "env_step_count": self.env_step_count,
            "policy_eligible_decision_count": self.policy_decision_count,
            "warmup_step_count": self.warmup_step_count,
            "warmup_hold_count": self.warmup_step_count,
            "warmup_replay_excluded_count": self.warmup_step_count,
            "action_counts": action_counts,
            "action_rates": {
                name: {
                    "numerator": action_counts[name],
                    "denominator": self.policy_decision_count,
                    "aggregation_unit": "policy_eligible_decision",
                    "value": (
                        action_counts[name] / self.policy_decision_count
                        if self.policy_decision_count
                        else None
                    ),
                }
                for name in VALID_ACTIONS
            },
            "action_by_position_before_counts": dict(self.action_position),
            "position_occupancy_step_counts": dict(self.position_occupancy),
            "entry_counts": dict(self.entry_counts),
            "close_counts": dict(self.close_counts),
            "completed_holding_time_seconds": holding,
            "timestamp_completed_holding_time_seconds": _summary(
                self.timestamp_completed_holding_seconds
            ),
            "timestamp_completed_holding_values_seconds": list(
                self.timestamp_completed_holding_seconds
            ),
            "completed_trade_count": int(sum(self.close_counts.values())),
            "normal_completed_trade_count": self.normal_completed_trade_count,
            "normal_close_count": self.normal_close_count,
            "five_second_close_rate": _ratio_metric(
                self.normal_close_within_5s_count,
                self.normal_completed_trade_count,
                "normal_completed_trade",
            ),
            "close_to_next_entry_seconds": _summary(
                self.close_to_next_entry_seconds
            ),
            "close_to_next_entry_values_seconds": list(
                self.close_to_next_entry_seconds
            ),
            "close_to_reentry_within_5s_rate": _ratio_metric(
                self.normal_close_followed_by_reentry_within_5s_count,
                self.normal_close_count,
                "successful_normal_close",
            ),
            "timestamp_holding_clock_semantic": (
                "decision_timestamp_difference; equivalent to effective execution "
                "timestamp difference under fixed deterministic latency"
            ),
            "full_episode_normalized_reward": self.full_normalized_reward,
            "full_episode_raw_yen_reward": self.full_raw_yen_reward,
            "warmup_normalized_reward": self.warmup_normalized_reward,
            "warmup_raw_yen_reward": self.warmup_raw_yen_reward,
            "policy_phase_normalized_reward": self.policy_normalized_reward,
            "policy_phase_raw_yen_reward": self.policy_raw_yen_reward,
            "reward_observation_valid_count": self.reward_observation_valid_count,
            "reward_observation_missing_count": self.reward_observation_missing_count,
            "final_cash_pnl": self.final_cash_pnl,
            "final_equity": self.final_equity,
            "warmup_by_session": {
                key: dict(value) for key, value in self.warmup_by_session.items()
            },
            "first_common_policy_eligible_timestamp_by_session": dict(
                self.first_policy_timestamp_by_session
            ),
            "session_aggregations": {
                key: dict(value) for key, value in self.session_aggregations.items()
            },
            "execution_failure_by_root_reason": dict(self.execution_failure_reasons),
            "valuation_failure_by_root_reason": dict(self.valuation_failure_reasons),
            "forced_exit_count": self.forced_exit_count,
            "forced_exit_failure_count": self.forced_exit_failure_count,
            "invalid_action_count": self.invalid_action_count,
            "review_required_count": self.review_required_count,
            "training_valid": self.training_valid,
        }
        if self.closingrisk_contract_active:
            result.update(
                {
                    "risk_management_step_count": self.risk_management_step_count,
                    "entry_cutoff_active_step_count": self.entry_cutoff_active_step_count,
                    "entry_cutoff_block_count": self.entry_cutoff_blocked_action_slot_count,
                    "entry_cutoff_block_rate": _ratio_metric(
                        self.entry_cutoff_blocked_action_slot_count,
                        self.entry_cutoff_env_valid_action_slot_count,
                        "env_valid_action_slot_during_entry_cutoff",
                    ),
                    "entry_cutoff_blocked_action_slot_count": self.entry_cutoff_blocked_action_slot_count,
                    "entry_cutoff_env_valid_action_slot_count": self.entry_cutoff_env_valid_action_slot_count,
                    "entry_cutoff_blocked_by_timestamp": dict(
                        self.entry_cutoff_blocked_by_timestamp
                    ),
                    "position_at_15_15_before": self.position_at_1515_before,
                    "position_at_15_15_after": self.position_at_1515_after,
                    "position_at_15_20_before": self.position_at_1520_before,
                    "position_at_15_20_after": self.position_at_1520_after,
                    "forced_liquidation_stage_counts": dict(
                        self.forced_liquidation_stages
                    ),
                    "forced_liquidation_resolution_delay_seconds": _summary(
                        self.forced_liquidation_delays
                    ),
                    "forced_liquidation_resolution_delay_values_seconds": list(
                        self.forced_liquidation_delays
                    ),
                    "closing_auction_use_count": self.closing_auction_use_count,
                    "forced_liquidation_unresolved_count": self.forced_liquidation_unresolved_count,
                }
            )
        if self.latency_contract_id is not None:
            result["latency_diagnostics"] = self._finalize_latency()
        return result

    def _record_holding(self, info: dict[str, Any]) -> None:
        value = info.get("completed_holding_time_seconds")
        if value is None:
            value = info.get("holding_time_seconds_before")
        parsed = _optional_finite(value)
        if parsed is not None and parsed >= 0:
            self.completed_holding_seconds.append(parsed)

    def _record_entry_timestamp(self, timestamp: datetime) -> None:
        if self._open_entry_timestamp is not None:
            raise ValueError("successful entry observed while a trade is already open")
        if self._last_normal_close_timestamp is not None:
            elapsed = (timestamp - self._last_normal_close_timestamp).total_seconds()
            if elapsed < 0:
                raise ValueError("entry timestamp precedes prior normal close")
            self.close_to_next_entry_seconds.append(float(elapsed))
            self.normal_close_followed_by_reentry_within_5s_count += int(
                elapsed <= 5.0
            )
            self._last_normal_close_timestamp = None
        self._open_entry_timestamp = timestamp

    def _record_close_timestamp(
        self, timestamp: datetime, info: dict[str, Any]
    ) -> None:
        if self._open_entry_timestamp is None:
            raise ValueError("successful close observed without an open entry timestamp")
        elapsed = (timestamp - self._open_entry_timestamp).total_seconds()
        if elapsed < 0:
            raise ValueError("close timestamp precedes entry timestamp")
        forced = bool(info.get("is_forced_exit", False))
        if not forced:
            self.timestamp_completed_holding_seconds.append(float(elapsed))
            self.normal_completed_trade_count += 1
            self.normal_close_count += 1
            self.normal_close_within_5s_count += int(elapsed <= 5.0)
            self._last_normal_close_timestamp = timestamp
        self._open_entry_timestamp = None

    def _effective_close_timestamp(
        self, decision_timestamp: str, info: dict[str, Any]
    ) -> datetime:
        if bool(info.get("is_forced_exit", False)):
            resolution = info.get("forced_liquidation_resolution_timestamp")
            if resolution is not None:
                return _parse_iso_timestamp(str(resolution))
        return _parse_iso_timestamp(decision_timestamp)

    def _observe_closingrisk(
        self,
        *,
        timestamp: str,
        position_before: str,
        position_after: str,
        info: dict[str, Any],
    ) -> None:
        parsed = _parse_iso_timestamp(timestamp)
        clock = parsed.time()
        if clock.hour == 15 and clock.minute == 15 and clock.second == 0:
            self.position_at_1515_before = position_before
            self.position_at_1515_after = position_after
        if clock.hour == 15 and clock.minute == 20 and clock.second == 0:
            self.position_at_1520_before = position_before
            self.position_at_1520_after = position_after
        if bool(info.get("entry_cutoff_active", False)):
            self.entry_cutoff_active_step_count += 1
            blocked = int(
                info.get("entry_cutoff_blocked_action_slot_count", 0) or 0
            )
            denominator = int(info.get("env_valid_action_slot_count", 0) or 0)
            self.entry_cutoff_blocked_action_slot_count += blocked
            self.entry_cutoff_env_valid_action_slot_count += denominator
            self.entry_cutoff_blocked_by_timestamp[
                parsed.isoformat()
            ] += blocked
        stage = info.get("forced_liquidation_stage")
        if stage is not None:
            self.forced_liquidation_stages[str(stage)] += 1
            delay = _optional_finite(
                info.get("forced_liquidation_delay_seconds")
            )
            if delay is not None and delay >= 0:
                self.forced_liquidation_delays.append(delay)
            self.closing_auction_use_count += int(
                bool(info.get("closing_auction_used", False))
            )
            self.forced_liquidation_unresolved_count += int(
                str(stage) == "unresolved"
            )

    def _observe_latency(self, *, info: dict[str, Any], session: str) -> None:
        contract_id = info.get("execution_contract_id")
        if contract_id is None:
            return
        if self.latency_contract_id is None:
            self.latency_contract_id = str(contract_id)
        elif self.latency_contract_id != str(contract_id):
            raise ValueError("execution latency contract changed within episode")
        routing_failed = bool(info.get("execution_routing_failed", False))
        if routing_failed:
            reason = str(
                info.get("execution_routing_failure_reason") or "unspecified"
            )
            self.latency_routing_failure_reasons[reason] += 1
            return
        snapshot_count = int(info.get("execution_snapshot_count", 0))
        if snapshot_count != 1:
            raise ValueError("valid latency step must expose one execution snapshot")
        self.latency_snapshot_count += 1
        advanced = info.get("execution_snapshot_advanced")
        same = info.get("execution_snapshot_same_as_decision")
        if type(advanced) is not bool or type(same) is not bool or advanced == same:
            raise ValueError("latency advanced/same flags are invalid")
        self.latency_snapshot_advanced_count += int(advanced)
        self.latency_snapshot_same_count += int(same)
        age = _optional_finite_strict(
            info.get("execution_snapshot_age_seconds"),
            "execution_snapshot_age_seconds",
        )
        if age is None or age < 0:
            raise ValueError("latency snapshot age must be nonnegative and present")
        self.latency_snapshot_ages.append(age)
        if bool(info.get("is_forced_exit", False)):
            self.latency_forced_snapshot_ages.append(age)
        self.latency_snapshot_by_session[session] += 1
        band = str(info.get("latency_time_band") or "missing")
        self.latency_snapshot_by_time_band[band] += 1
        if bool(info.get("latency_policy_selected_execution", False)):
            self.latency_actual_execution_count += 1
            side = str(info.get("latency_policy_selected_action_side") or "missing")
            delta = _optional_finite(
                info.get("delayed_vs_decision_execution_price_delta_bps")
            )
            if side not in {"Buy", "Sell"} or delta is None:
                self.latency_actual_execution_delta_missing_count += 1
            else:
                self.latency_actual_delta_by_side[side].append(delta)
                sign = "positive" if delta > 0 else "negative" if delta < 0 else "zero"
                self.latency_actual_delta_sign_by_side[f"{side}|{sign}"] += 1

    def _finalize_latency(self) -> dict[str, Any]:
        count = self.latency_snapshot_count
        ages = self.latency_snapshot_ages
        return {
            "execution_contract_id": self.latency_contract_id,
            "execution_snapshot_count": count,
            "execution_snapshot_advanced_rate": _ratio_metric(
                self.latency_snapshot_advanced_count, count, "decision_step"
            ),
            "execution_snapshot_same_as_decision_rate": _ratio_metric(
                self.latency_snapshot_same_count, count, "decision_step"
            ),
            "execution_snapshot_age_seconds": _latency_distribution(ages),
            "execution_snapshot_age_eq_0_count": sum(value == 0 for value in ages),
            "execution_snapshot_age_le_1s_count": sum(value <= 1 for value in ages),
            "execution_snapshot_age_gt_5s_count": sum(value > 5 for value in ages),
            "execution_snapshot_age_gt_10s_count": sum(value > 10 for value in ages),
            "forced_exit_snapshot_age_seconds": _latency_distribution(
                self.latency_forced_snapshot_ages
            ),
            "routing_failure_by_reason": dict(
                self.latency_routing_failure_reasons
            ),
            "routing_failure_count": int(
                sum(self.latency_routing_failure_reasons.values())
            ),
            "snapshot_count_by_session": dict(self.latency_snapshot_by_session),
            "snapshot_count_by_time_band": dict(
                self.latency_snapshot_by_time_band
            ),
            "actual_policy_execution_count": self.latency_actual_execution_count,
            "actual_policy_execution_delta_missing_count": (
                self.latency_actual_execution_delta_missing_count
            ),
            "actual_policy_execution_delta_bps_by_side": {
                side: _latency_distribution(
                    self.latency_actual_delta_by_side.get(side, [])
                )
                for side in ("Buy", "Sell")
            },
            "actual_policy_execution_delta_sign_by_side": dict(
                self.latency_actual_delta_sign_by_side
            ),
        }


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": _quantile(ordered, 0.50),
        "p95": _quantile(ordered, 0.95),
        "max": ordered[-1],
    }


def _latency_distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            key: (0 if key == "count" else None)
            for key in (
                "count",
                "mean",
                "std",
                "min",
                "p01",
                "p05",
                "p50",
                "p90",
                "p95",
                "p99",
                "max",
            )
        }
    ordered = sorted(values)
    mean = sum(ordered) / len(ordered)
    return {
        "count": len(ordered),
        "mean": mean,
        "std": math.sqrt(
            sum((value - mean) ** 2 for value in ordered) / len(ordered)
        ),
        "min": ordered[0],
        "p01": _quantile(ordered, 0.01),
        "p05": _quantile(ordered, 0.05),
        "p50": _quantile(ordered, 0.50),
        "p90": _quantile(ordered, 0.90),
        "p95": _quantile(ordered, 0.95),
        "p99": _quantile(ordered, 0.99),
        "max": ordered[-1],
    }


def _ratio_metric(
    numerator: int, denominator: int, aggregation_unit: str
) -> dict[str, Any]:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "aggregation_unit": str(aggregation_unit),
        "value": numerator / denominator if denominator else None,
    }


def _parse_iso_timestamp(value: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError("timestamp must be ISO-8601") from exc
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return timestamp


def _quantile(values: list[float], probability: float) -> float:
    index = (len(values) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_finite_strict(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, name)


__all__ = [
    "E0EpisodeAccountingV01",
    "POLICY_STATUS_NO_READY",
    "RepresentationActionDecisionV01",
    "NEXT_REPRESENTATION_UNREADY_EXCLUSION_REASON",
    "WARMUP_CONTRACT_ID",
    "exclude_policy_transition_for_next_readiness_v01",
    "select_representation_action_v01",
]
