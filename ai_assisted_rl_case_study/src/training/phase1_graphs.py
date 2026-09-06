"""I12 Phase-I plotting blocks; original numerators, denominators and axes retained."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from training.public_graphs import save_figure
SEEDS = (20260815, 20260821, 20260822)
def plot_checkpoint_reward(summary: pd.DataFrame, output: Path) -> None:
    source = summary[(summary.aggregation == "seed") | (summary.seed == "3_seed_mean")].copy()
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    colors = {20260815: "#1f77b4", 20260821: "#ff7f0e", 20260822: "#2ca02c"}
    for cohort, linestyle in (("baseline", "--"), ("candidate", "-")):
        for seed in SEEDS:
            part = source[(source.cohort == cohort) & (source.seed == str(seed))].sort_values("checkpoint")
            ax.plot(part.checkpoint, part["mean"], marker="o", linestyle=linestyle, color=colors[seed], alpha=0.72, label=f"{cohort} {seed}")
        mean = source[(source.cohort == cohort) & (source.seed == "3_seed_mean")].sort_values("checkpoint")
        ax.plot(mean.checkpoint, mean["mean"], linewidth=3.0, color="black" if cohort == "baseline" else "#d62728", linestyle=linestyle, label=f"{cohort} 3-seed mean")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel("Optimizer update")
    ax.set_ylabel("Mean episode cumulative normalized reward")
    ax.set_title("DDQN development reward progression (fixed 48-episode sentinel)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save_figure(fig, output / "reward_progression_by_checkpoint.png", dpi=160)
    plt.close(fig)

def numeric_time(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[frame.time_since_close_seconds != ">120"].copy()
    result["time_since_close_seconds"] = pd.to_numeric(result.time_since_close_seconds)
    return result

def plot_time_since_close(frame: pd.DataFrame, refs: pd.DataFrame, output: Path) -> None:
    numeric = numeric_time(frame)
    ddqn = numeric[(numeric.condition == "H1_DDQN") & (numeric.seed == "all")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharex=True, sharey=True)
    for ax, side in zip(axes, ("Long", "Short")):
        for cohort, style, color in (("baseline", "--", "#555555"), ("candidate", "-", "#1f77b4")):
            part = ddqn[(ddqn.cohort == cohort) & (ddqn.close_side == side) & (ddqn.time_since_close_seconds <= 120) & (ddqn.decision_count > 0)]
            ax.plot(part.time_since_close_seconds, part.entry_rate, marker="o", markersize=3, linestyle=style, color=color, label=f"{cohort} actual Entry")
        ref = refs[(refs.cohort == "candidate") & (refs.condition == "H1_DDQN") & (refs.seed == "all")].iloc[0]
        ax.axhline(float(ref.entry_rate), color="#2ca02c", linestyle=":", label="candidate continuous-Flat reference")
        ax.set_title(f"After successful {side} Close")
        ax.set_xlabel("Seconds since Close (same session)")
        ax.set_xlim(5, 120)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Selected Entry rate")
    axes[1].legend(fontsize=8)
    fig.suptitle("DDQN selected action by time since successful normal Close")
    fig.tight_layout()
    save_figure(fig, output / "ddqn_entry_rate_by_time_since_close.png", dpi=160)
    plt.close(fig)

