"""
Visualization — Plotting utilities for training analysis and evaluation.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def plot_training_curves(rewards, rates, defaults, save_path="training_curves.png"):
    """Plot episode rewards, avg rates, and default rates over training."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Training Progress", fontsize=14, fontweight="bold")

    axes[0].plot(rewards, alpha=0.3, color="#4a90d9")
    if len(rewards) > 20:
        window = max(len(rewards) // 20, 5)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
        axes[0].plot(range(window-1, len(rewards)), smoothed, color="#1a5276", linewidth=2)
    axes[0].set_ylabel("Episode Reward")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rates, alpha=0.3, color="#27ae60")
    if len(rates) > 20:
        window = max(len(rates) // 20, 5)
        smoothed = np.convolve(rates, np.ones(window)/window, mode="valid")
        axes[1].plot(range(window-1, len(rates)), smoothed, color="#1e8449", linewidth=2)
    axes[1].set_ylabel("Avg Interest Rate")
    axes[1].axhline(y=0.15, color="red", linestyle="--", alpha=0.5, label="Baseline")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(defaults, alpha=0.3, color="#e74c3c")
    if len(defaults) > 20:
        window = max(len(defaults) // 20, 5)
        smoothed = np.convolve(defaults, np.ones(window)/window, mode="valid")
        axes[2].plot(range(window-1, len(defaults)), smoothed, color="#922b21", linewidth=2)
    axes[2].set_ylabel("Default Rate")
    axes[2].set_xlabel("Episode")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_fairness_dashboard(fairness_report, save_path="fairness_dashboard.png"):
    """Plot fairness metrics: SPD and DI ratio with compliance bands."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fairness Evaluation Dashboard", fontsize=14, fontweight="bold")

    demo = fairness_report.get("demographic", {})

    # SPD bar
    spd = demo.get("spd", 0)
    color = "#27ae60" if abs(spd) <= 0.1 else "#e74c3c"
    axes[0].bar(["Demographic SPD"], [spd], color=color, edgecolor="black", width=0.4)
    axes[0].axhline(y=0.1, color="red", linestyle="--", alpha=0.5, label="Upper bound")
    axes[0].axhline(y=-0.1, color="red", linestyle="--", alpha=0.5, label="Lower bound")
    axes[0].axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    axes[0].set_ylabel("Statistical Parity Difference")
    axes[0].set_ylim(-0.3, 0.3)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # DI ratio bar
    di = demo.get("di_ratio", 1.0)
    color = "#27ae60" if 0.8 <= di <= 1.25 else "#e74c3c"
    axes[1].bar(["Demographic DI"], [di], color=color, edgecolor="black", width=0.4)
    axes[1].axhline(y=0.8, color="red", linestyle="--", alpha=0.5, label="Min (0.8)")
    axes[1].axhline(y=1.25, color="red", linestyle="--", alpha=0.5, label="Max (1.25)")
    axes[1].axhline(y=1.0, color="gray", linestyle="-", alpha=0.3)
    axes[1].set_ylabel("Disparate Impact Ratio")
    axes[1].set_ylim(0.0, 2.0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_curriculum_weights(weight_history, save_path="curriculum_weights.png"):
    """Plot the curriculum learning weight schedule over training."""
    steps = [w.get("step", i) for i, w in enumerate(weight_history)]
    w1s = [w["w1"] for w in weight_history]
    w2s = [w["w2"] for w in weight_history]
    w3s = [w["w3"] for w in weight_history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, w1s, label="w1 (Interest)", color="#3498db", linewidth=2)
    ax.plot(steps, w2s, label="w2 (Default Risk)", color="#e74c3c", linewidth=2)
    ax.plot(steps, w3s, label="w3 (Client Utility)", color="#27ae60", linewidth=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Weight Value")
    ax.set_title("Curriculum Learning: Reward Weight Schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_rate_distribution(rates, groups, save_path="rate_distribution.png"):
    """Plot interest rate distributions by demographic group."""
    fig, ax = plt.subplots(figsize=(10, 5))
    rates, groups = np.asarray(rates), np.asarray(groups)

    majority_rates = rates[groups == 0]
    minority_rates = rates[groups == 1]

    ax.hist(majority_rates, bins=30, alpha=0.6, label="Majority", color="#3498db", edgecolor="black")
    ax.hist(minority_rates, bins=30, alpha=0.6, label="Minority", color="#e67e22", edgecolor="black")
    ax.axvline(x=np.mean(majority_rates), color="#2c3e50", linestyle="--", linewidth=2, label=f"Maj. mean: {np.mean(majority_rates):.3f}")
    ax.axvline(x=np.mean(minority_rates), color="#d35400", linestyle="--", linewidth=2, label=f"Min. mean: {np.mean(minority_rates):.3f}")
    ax.set_xlabel("Interest Rate")
    ax.set_ylabel("Count")
    ax.set_title("Interest Rate Distribution by Demographic Group")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path
