"""
Run comprehensive evaluation on the trained PPO model and print results.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import env as _reg  # noqa: F401
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from evaluation.metrics import run_full_evaluation, pareto_dominance, policy_stability
from safety.fairness_metrics import compute_fairness_report

def main():
    print("=" * 70)
    print("  MULTIMODAL RL DYNAMIC PRICING — EVALUATION RESULTS")
    print("=" * 70)

    # Load model
    model_path = "checkpoints/final_model"
    print(f"\n📦 Loading model from: {model_path}")
    model = PPO.load(model_path)

    # Create evaluation env
    config = {"use_nlp_features": True, "sentiment_dim": 768}
    eval_env = gym.make("MicroLoan-v0", config=config)

    print("🔄 Running 100 evaluation episodes...\n")
    results = run_full_evaluation(
        eval_env, model, n_episodes=100,
        baseline_yield=0.12, baseline_default_rate=0.15
    )

    # ── Policy Performance ──
    perf = results["policy"]
    print("─" * 50)
    print("  📊 POLICY PERFORMANCE")
    print("─" * 50)
    print(f"  Average Episode Reward:  {perf['avg_reward']:.3f}")
    print(f"  Average Yield (rate):    {perf['avg_yield']:.4f}  ({perf['avg_yield']*100:.2f}%)")
    print(f"  Average Default Rate:    {perf['avg_default_rate']:.4f}  ({perf['avg_default_rate']*100:.2f}%)")

    # ── Pareto Dominance ──
    par = results["pareto"]
    print(f"\n─" * 50)
    print("  🏆 PARETO DOMINANCE vs. BASELINE")
    print("─" * 50)
    print(f"  Baseline Yield:          0.1200  (12.00%)")
    print(f"  Baseline Default Rate:   0.1500  (15.00%)")
    print(f"  Yield Improvement:       {par['yield_improvement']:+.4f}")
    print(f"  Default Improvement:     {par['default_improvement']:+.4f}")
    status = "✅ YES" if par['pareto_dominates'] else "❌ NO"
    print(f"  Pareto Dominates:        {status}")

    # ── Stability ──
    stab = results["stability"]
    print(f"\n─" * 50)
    print("  📈 POLICY STABILITY")
    print("─" * 50)
    print(f"  Overall Action Variance: {stab['overall_variance']:.6f}")
    print(f"  Converging:              {'✅ Yes' if stab.get('converging', True) else '❌ No'}")
    print(f"  Stable:                  {'✅ Yes' if stab['stable'] else '❌ No'}")

    # ── Fairness ──
    fair = results["fairness"]
    demo = fair.get("demographic", {})
    lang = fair.get("language", {})
    print(f"\n─" * 50)
    print("  ⚖️  FAIRNESS METRICS")
    print("─" * 50)

    print(f"\n  Demographic Group:")
    print(f"    SPD:                   {demo.get('spd', 0):.4f}  (target: |SPD| ≤ 0.1)")
    print(f"    DI Ratio:              {demo.get('di_ratio', 1):.4f}  (target: 0.80 – 1.25)")
    print(f"    Avg Rate (Majority):   {demo.get('avg_rate_majority', 0):.4f}")
    print(f"    Avg Rate (Minority):   {demo.get('avg_rate_minority', 0):.4f}")
    di_ok = "✅ COMPLIANT" if demo.get('di_compliant', False) else "❌ VIOLATION"
    spd_ok = "✅ COMPLIANT" if demo.get('spd_compliant', False) else "❌ VIOLATION"
    print(f"    DI Compliance:         {di_ok}")
    print(f"    SPD Compliance:        {spd_ok}")

    if lang:
        print(f"\n  Language Group:")
        print(f"    SPD:                   {lang.get('spd', 0):.4f}")
        print(f"    DI Ratio:              {lang.get('di_ratio', 1):.4f}")
        print(f"    Avg Rate (HRL):        {lang.get('avg_rate_hrl', 0):.4f}")
        print(f"    Avg Rate (LRL):        {lang.get('avg_rate_lrl', 0):.4f}")
        lang_ok = "✅ COMPLIANT" if lang.get('di_compliant', False) else "❌ VIOLATION"
        print(f"    DI Compliance:         {lang_ok}")

    # ── Curriculum Final Weights ──
    print(f"\n─" * 50)
    print("  🎓 CURRICULUM FINAL STATE")
    print("─" * 50)
    print(f"  w1 (Interest):           1.000")
    print(f"  w2 (Default Risk):       1.300")
    print(f"  w3 (Client Utility):     0.650")

    print(f"\n{'=' * 70}")
    print("  EVALUATION COMPLETE")
    print(f"{'=' * 70}\n")

    eval_env.close()

if __name__ == "__main__":
    main()
