"""I12 historical graph rendering, later report tooling, not training code.
描画式・集計・軸はhistorical finalizerから保持。保存先だけ呼出し側が指定する。
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from training.public_graphs import save_graph
OUT = None
def require(condition, message):
    if not condition: raise RuntimeError(message)
def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')

VARIANT_ORDER = ("B0", "L1", "L2", "L3")

DEPTH = {"B0": 0, "L1": 1, "L2": 2, "L3": 3}

WINDOW_DROP = {"B0": None, "L1": 10, "L2": 15, "L3": 20}

STYLE = {
    "B0": ("#555555", "--", "o"),
    "L1": ("#1f77b4", "-", "s"),
    "L2": ("#ff7f0e", "-", "^"),
    "L3": ("#2ca02c", "-", "D"),
}

def create_graphs(
    memory: pd.DataFrame,
    seed: pd.DataFrame,
    checkpoint: pd.DataFrame,
    behavior: pd.DataFrame,
    learning: pd.DataFrame,
    tail: pd.DataFrame,
    cdf: pd.DataFrame,
    time_combined: pd.DataFrame,
    window: pd.DataFrame,
    history: pd.DataFrame,
) -> dict[str, Any]:
    manifest: list[dict[str, Any]] = []
    seed_mean = seed[(seed.metric == "mean_normalized_pnl") & (seed.seed.astype(str) != "all")].copy()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(seed_mean))
    ax.bar(x, seed_mean.value, color=[STYLE[value][0] for value in seed_mean.variant])
    ax.set_xticks(x, [f"{row.variant}\n{row.seed}" for row in seed_mean.itertuples()], rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.8); ax.set_ylabel("Mean normalized PnL / episode")
    save_graph(manifest, graph_id="G01", figure=fig, source=seed_mean, purpose="B0/L1/L2/L3 final normalized PnL by seed", axis={"x": "variant and seed", "y": "mean normalized PnL"}, unit="normalized PnL/episode", series=VARIANT_ORDER, denominator="144 fixed development episodes per variant/seed", limitations="development diagnostic; update_2000 fixed, no checkpoint selection")

    for graph_id, metric, purpose in (("G02", "mean_normalized_pnl", "Mean normalized PnL by explicit-memory depth"), ("G03", "median_normalized_pnl", "Median normalized PnL by explicit-memory depth")):
        source = memory[memory.metric == metric].sort_values("memory_depth")
        fig, ax = plt.subplots(figsize=(7, 5)); ax.plot(source.memory_depth, source.value, marker="o"); ax.axhline(0, color="black", linewidth=0.8); ax.set_xticks(range(4), VARIANT_ORDER); ax.set_ylabel(metric)
        save_graph(manifest, graph_id=graph_id, figure=fig, source=source, purpose=purpose, axis={"x": "explicit-memory depth", "y": metric}, unit="normalized PnL/episode", series=VARIANT_ORDER, denominator="432 outcomes per variant", limitations="depth is descriptive; variants are independently trained")

    fig, ax = plt.subplots(figsize=(10, 5.5)); metrics = ["p05_normalized_pnl", "p01_normalized_pnl", "worst_5pct_mean_normalized_pnl", "maximum_loss_normalized_pnl"]
    x = np.arange(len(metrics)); width = 0.2
    for index, variant in enumerate(VARIANT_ORDER):
        part = tail[tail.variant == variant].set_index("metric").loc[metrics]
        ax.bar(x + (index - 1.5) * width, part.value, width, label=variant, color=STYLE[variant][0])
    ax.set_xticks(x, ["p05", "p01", "worst 5% mean", "maximum loss"]); ax.axhline(0, color="black", linewidth=0.8); ax.legend()
    save_graph(manifest, graph_id="G04", figure=fig, source=tail, purpose="Tail-risk comparison", axis={"x": "tail metric", "y": "normalized PnL"}, unit="normalized PnL/episode", series=VARIANT_ORDER, denominator="432 outcomes per variant; worst-tail count 22", limitations="more negative is worse; common axis")

    cp = checkpoint[(checkpoint.metric == "mean_normalized_pnl") & (checkpoint.seed.astype(str) == "all")]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for variant in VARIANT_ORDER:
        part = cp[cp.variant == variant].sort_values("checkpoint"); color, style, marker = STYLE[variant]; ax.plot(part.checkpoint, part.value, color=color, linestyle=style, marker=marker, label=variant)
    ax.axhline(0, color="black", linewidth=0.8); ax.legend(); ax.set_xlabel("Optimizer update"); ax.set_ylabel("Mean normalized PnL")
    save_graph(manifest, graph_id="G05", figure=fig, source=cp, purpose="Checkpoint reward/PnL progression", axis={"x": "optimizer update", "y": "3-seed mean normalized PnL"}, unit="normalized PnL/episode", series=VARIANT_ORDER, denominator="fixed 48-episode 2026-06-30 sentinel per seed/checkpoint for B0/L1 comparability", limitations="L2/L3 also retain full 144 diagnostics; update_2000 remains primary", seed_checkpoint="3 seeds / updates 0,250,500,1000,1500,2000")

    for graph_id, field, purpose in (("G06", "loss_mean", "Huber loss progression"), ("G07", "td_abs_mean", "Absolute TD-error progression")):
        source = learning.groupby(["variant", "update_end"], as_index=False)[field].mean()
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for variant in VARIANT_ORDER:
            part = source[source.variant == variant]; color, style, _ = STYLE[variant]; ax.plot(part.update_end, part[field], color=color, linestyle=style, label=variant)
        ax.set_yscale("log"); ax.legend(); ax.set_xlabel("Optimizer update"); ax.set_ylabel(field)
        save_graph(manifest, graph_id=graph_id, figure=fig, source=source, purpose=purpose, axis={"x": "optimizer update", "y": field + " (log scale)"}, unit="loss/Q units", series=VARIANT_ORDER, denominator="128 replay transitions/update; 3 seeds", limitations="identical 50-update smoothing")

    entries = behavior[(behavior.metric == "entries_per_episode") & (behavior.seed.astype(str) == "all")]
    fig, ax = plt.subplots(figsize=(7, 5)); ax.bar(entries.variant, entries.value, color=[STYLE[x][0] for x in entries.variant]); ax.set_ylabel("Entries / episode")
    save_graph(manifest, graph_id="G08", figure=fig, source=entries, purpose="Entries per episode", axis={"x": "variant", "y": "entries/episode"}, unit="events/episode", series=VARIANT_ORDER, denominator="432 episodes per variant", limitations="policy-dependent event count")

    holding = behavior[(behavior.metric.isin(["mean_holding_seconds", "median_holding_seconds"])) & (behavior.seed.astype(str) == "all")]
    pivot = holding.pivot(index="variant", columns="metric", values="value").reindex(VARIANT_ORDER)
    fig, ax = plt.subplots(figsize=(8, 5)); x = np.arange(4); ax.bar(x - .18, pivot.mean_holding_seconds, .36, label="mean"); ax.bar(x + .18, pivot.median_holding_seconds, .36, label="median"); ax.set_xticks(x, VARIANT_ORDER); ax.legend(); ax.set_ylabel("Seconds")
    save_graph(manifest, graph_id="G09", figure=fig, source=holding, purpose="Mean and median holding time", axis={"x": "variant", "y": "holding seconds"}, unit="seconds", series=("mean", "median"), denominator="timestamp-completed trades; denominator in source", limitations="outcome-dependent holding denominator")

    five = behavior[(behavior.metric == "five_second_close_rate") & (behavior.seed.astype(str) == "all")]
    fig, ax = plt.subplots(figsize=(7, 5)); ax.bar(five.variant, five.value, color=[STYLE[x][0] for x in five.variant]); ax.set_ylim(0, 1); ax.set_ylabel("5-second Close rate")
    save_graph(manifest, graph_id="G10", figure=fig, source=five, purpose="Five-second Close rate", axis={"x": "variant", "y": "rate"}, unit="fraction", series=VARIANT_ORDER, denominator="successful normal completed trades; denominator in source", limitations="policy-dependent denominators")

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in VARIANT_ORDER:
        part = cdf[cdf.variant == variant]; color, style, marker = STYLE[variant]; ax.plot(part.threshold_seconds, part.rate, color=color, linestyle=style, marker=marker, label=variant)
    ax.set_ylim(0, 1); ax.legend(); ax.set_xlabel("Seconds since Close"); ax.set_ylabel("Cumulative re-entry fraction")
    save_graph(manifest, graph_id="G11", figure=fig, source=cdf, purpose="Close-to-re-entry CDF", axis={"x": "threshold seconds", "y": "CDF"}, unit="fraction", series=VARIANT_ORDER, denominator="successful normal Close events", limitations="episode-continuous legacy definition retained for exact B0/L1 compatibility")

    for graph_id, field, purpose in (("G12", "entry_rate", "Entry rate by same-session time since Close"), ("G13", "same_side_reentry_rate", "Same-side re-entry rate by same-session time since Close"), ("G14", "reversal_rate", "Reversal rate by same-session time since Close"), ("G15", "hold_rate", "Hold rate by same-session time since Close")):
        source = time_combined[time_combined.time_since_close_seconds <= 120].copy()
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for variant in VARIANT_ORDER:
            part = source[source.variant == variant]; color, style, marker = STYLE[variant]; ax.plot(part.time_since_close_seconds, part[field], color=color, linestyle=style, marker=marker, markersize=3, label=variant)
        if field.endswith("rate"): ax.set_ylim(0, 1)
        ax.axhline(0, color="black", linewidth=0.6); ax.legend(); ax.set_xlabel("Same-session seconds since successful normal Close"); ax.set_ylabel(field)
        save_graph(manifest, graph_id=graph_id, figure=fig, source=source, purpose=purpose, axis={"x": "same-session seconds since Close", "y": field}, unit="fraction" if field.endswith("rate") else "Q units", series=VARIANT_ORDER, denominator="Flat ordinary decisions before next successful Entry; Long/Short pooled", limitations="risk-management decisions excluded; changing risk set over elapsed time")

    drop_rows = []
    for variant in ("L1", "L2", "L3"):
        drop = int(WINDOW_DROP[variant])
        for elapsed, label in ((drop - 5, "last_present"), (drop, "first_dropped")):
            row = window[(window.variant == variant) & (window.time_since_close_seconds == elapsed)].iloc[0]
            drop_rows.append({**row.to_dict(), "comparison_point": label})
    drop_source = pd.DataFrame(drop_rows)
    fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(3); width = .35
    present = drop_source[drop_source.comparison_point == "last_present"].set_index("variant").reindex(("L1", "L2", "L3"))
    dropped = drop_source[drop_source.comparison_point == "first_dropped"].set_index("variant").reindex(("L1", "L2", "L3"))
    ax.bar(x - width / 2, present.same_side_reentry_rate, width, label="last present"); ax.bar(x + width / 2, dropped.same_side_reentry_rate, width, label="first dropped"); ax.set_xticks(x, ("L1 +10s", "L2 +15s", "L3 +20s")); ax.set_ylim(0, 1); ax.legend(); ax.set_ylabel("Same-side re-entry rate")
    save_graph(manifest, graph_id="G16", figure=fig, source=drop_source, purpose="Window-drop re-entry comparison", axis={"x": "candidate expected window drop", "y": "same-side re-entry rate"}, unit="fraction", series=("last context-present decision", "first context-dropped decision"), denominator="same-session Flat decisions before next successful Entry", limitations="descriptive mechanism diagnostic, not causal proof")

    source = time_combined[time_combined.time_since_close_seconds <= 120].copy()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for variant in VARIANT_ORDER:
        part = source[source.variant == variant]; color, style, marker = STYLE[variant]; ax.plot(part.time_since_close_seconds, part.entry_advantage_mean, color=color, linestyle=style, marker=marker, markersize=3, label=variant)
    ax.axhline(0, color="black", linewidth=0.6); ax.legend(); ax.set_xlabel("Same-session seconds since successful normal Close"); ax.set_ylabel("entry_advantage_mean")
    save_graph(manifest, graph_id="G17", figure=fig, source=source, purpose="Q Entry-vs-Hold advantage by same-session time since Close", axis={"x": "same-session seconds since Close", "y": "entry_advantage_mean"}, unit="Q units", series=VARIANT_ORDER, denominator="Flat ordinary decisions before next successful Entry; Long/Short pooled", limitations="risk-management decisions excluded; changing risk set over elapsed time")

    action = behavior[(behavior.metric.isin(["action_rate_hold", "action_rate_buy", "action_rate_sell"])) & (behavior.seed.astype(str) == "all")]
    ap = action.pivot(index="variant", columns="metric", values="value").reindex(VARIANT_ORDER)
    fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(4); width = .25
    for index, metric in enumerate(("action_rate_hold", "action_rate_buy", "action_rate_sell")):
        ax.bar(x + (index - 1) * width, ap[metric], width, label=metric.removeprefix("action_rate_"))
    ax.set_xticks(x, VARIANT_ORDER); ax.set_ylim(0, 1); ax.legend()
    save_graph(manifest, graph_id="G18", figure=fig, source=action, purpose="Action distribution", axis={"x": "variant", "y": "policy action rate"}, unit="fraction", series=("Hold", "Buy", "Sell"), denominator="policy-eligible decisions", limitations="risk-management actions excluded by episode policy counts")

    position = behavior[(behavior.metric.str.startswith("position_")) & (behavior.seed.astype(str) == "all")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    for axis, pos in zip(axes, ("flat", "long", "short")):
        part = position[position.metric.str.startswith(f"position_{pos}_")]
        for index, variant in enumerate(VARIANT_ORDER):
            values = [float(part[(part.variant == variant) & (part.metric == f"position_{pos}_action_{action}_rate")].value.iloc[0]) for action in ("hold", "buy", "sell")]
            axis.plot(("Hold", "Buy", "Sell"), values, marker=STYLE[variant][2], color=STYLE[variant][0], label=variant)
        axis.set_title(pos.title()); axis.set_ylim(0, 1)
    axes[0].legend(); axes[0].set_ylabel("Conditional action rate")
    save_graph(manifest, graph_id="G19", figure=fig, source=position, purpose="Position-conditioned action distribution", axis={"x": "semantic action by position-before", "y": "conditional rate"}, unit="fraction", series=VARIANT_ORDER, denominator="position-before decision count; denominator in source", limitations="position occupancy differs by policy")

    hist = history[(history.population == "development_update_2000")].copy()
    hist_agg = hist.groupby(["variant", "lag"], as_index=False).agg(available_count=("available_count", "sum"), state_count=("state_count", "sum"), invariant_violations=("invariant_violations", "sum"), unexpected_cross_session_carry=("unexpected_cross_session_carry", "sum"))
    hist_agg["availability_rate"] = hist_agg.available_count / hist_agg.state_count
    fig, ax = plt.subplots(figsize=(8, 5)); labels = [f"{row.variant}-prev{row.lag}" for row in hist_agg.itertuples()]; ax.bar(labels, hist_agg.availability_rate); ax.set_ylim(0, 1.05); ax.set_ylabel("Availability rate"); ax.tick_params(axis="x", rotation=35)
    save_graph(manifest, graph_id="G20", figure=fig, source=hist_agg, purpose="History availability and invariant QC", axis={"x": "candidate history lag", "y": "availability rate"}, unit="fraction", series=("L1", "L2", "L3"), denominator="development state count; invariant/cross-session counts in source", limitations="B0 N/A because no explicit history")

    delta = seed_mean[seed_mean.variant.isin(["L2", "L3"])].copy()
    delta_long = pd.concat([
        delta.assign(comparator="B0", delta=delta.delta_vs_B0),
        delta.assign(comparator="L1", delta=delta.delta_vs_L1),
    ], ignore_index=True)
    fig, ax = plt.subplots(figsize=(10, 5)); labels = [f"{row.variant}-{row.seed}\nvs {row.comparator}" for row in delta_long.itertuples()]; ax.bar(labels, delta_long.delta, color=[STYLE[row.variant][0] for row in delta_long.itertuples()]); ax.axhline(0, color="black", linewidth=.8); ax.tick_params(axis="x", rotation=45); ax.set_ylabel("Mean normalized PnL delta")
    save_graph(manifest, graph_id="G21", figure=fig, source=delta_long, purpose="Seed delta comparison versus B0 and L1", axis={"x": "candidate/seed/comparator", "y": "paired mean delta"}, unit="normalized PnL/episode", series=("L2", "L3"), denominator="144 exactly paired episodes", limitations="positive favors candidate")

    summary_rows = []
    for variant in VARIANT_ORDER:
        summary_rows.append({
            "variant": variant,
            "memory_depth": DEPTH[variant],
            "mean_normalized_pnl": float(memory[(memory.variant == variant) & (memory.metric == "mean_normalized_pnl")].value.iloc[0]),
            "entries_per_episode": float(behavior[(behavior.variant == variant) & (behavior.metric == "entries_per_episode") & (behavior.seed.astype(str) == "all")].value.iloc[0]),
            "mean_holding_seconds": float(behavior[(behavior.variant == variant) & (behavior.metric == "mean_holding_seconds") & (behavior.seed.astype(str) == "all")].value.iloc[0]),
            "five_second_close_rate": float(behavior[(behavior.variant == variant) & (behavior.metric == "five_second_close_rate") & (behavior.seed.astype(str) == "all")].value.iloc[0]),
        })
    summary_source = pd.DataFrame(summary_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8)); axes[0].plot(summary_source.memory_depth, summary_source.mean_normalized_pnl, marker="o"); axes[0].axhline(0, color="black", linewidth=.8); axes[0].set_xticks(range(4), VARIANT_ORDER); axes[0].set_ylabel("Mean normalized PnL"); axes[1].plot(summary_source.memory_depth, summary_source.entries_per_episode, marker="o", label="entries/episode"); axes[1].plot(summary_source.memory_depth, summary_source.mean_holding_seconds, marker="s", label="mean holding seconds"); axes[1].set_xticks(range(4), VARIANT_ORDER); axes[1].legend()
    save_graph(manifest, graph_id="G22", figure=fig, source=summary_source, purpose="Memory-depth economic and behavior summary", axis={"x": "explicit-memory depth", "y": "economic/behavior metric"}, unit="mixed; separate panels", series=("mean normalized PnL", "entries/episode", "mean holding seconds"), denominator="432 outcomes/variant and policy-dependent behavior denominators", limitations="mixed metrics use separate panels; descriptive, not causal")

    require([item["graph_id"] for item in manifest] == [f"G{index:02d}" for index in range(1, 23)], "graph manifest order/count mismatch")
    payload = {
        "schema_id": "ddqn_lag2_lag3_history_e0_2_graph_manifest_v01",
        "status": "PASS",
        "required_graph_count": 22,
        "completed_graph_count": len(manifest),
        "graph_set_identity_sha256": canonical_sha256([item["graph_id"] for item in manifest]),
        "graphs": manifest,
    }
    write_json(OUT / "graph_manifest.json", payload)
    return payload
