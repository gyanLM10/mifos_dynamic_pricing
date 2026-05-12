"""
Comprehensive 4-Dimension Evaluation — Measures ALL metrics:
  1. NLP Subsystem: Macro F1-Score and Recall
  2. Environment Fidelity: K-S Test and MAE
  3. RL Policy Performance: Pareto Dominance and Stability
  4. Fairness: SPD and DI Ratio
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import env as _reg  # noqa: F401
import gymnasium as gym
from stable_baselines3 import PPO
from scipy import stats

from evaluation.metrics import (
    compute_nlp_metrics, ks_test_fidelity, mae_transition_fidelity,
    pareto_dominance, policy_stability,
)
from safety.fairness_metrics import (
    statistical_parity_difference, disparate_impact_ratio, compute_fairness_report,
)
from data.mifos_calibrator import MifosCalibrator
from nlp.sentiment_mapper import MockSentimentMapper

HEADER = "=" * 72
SECTION = "─" * 72


def print_header(title):
    print(f"\n{HEADER}")
    print(f"  {title}")
    print(HEADER)


def print_section(title):
    print(f"\n{SECTION}")
    print(f"  {title}")
    print(SECTION)


def main():
    print(HEADER)
    print("  MULTIMODAL RL — COMPREHENSIVE 4-DIMENSION EVALUATION")
    print(f"  All metrics measured across NLP, Fidelity, Policy, and Fairness")
    print(HEADER)

    # ===================================================================
    # DIMENSION 1: NLP SUBSYSTEM (Classification)
    # ===================================================================
    print_header("DIMENSION 1: NLP SUBSYSTEM (Classification)")
    print("  Evaluating Mock Sentiment Mapper risk classification accuracy")
    print("  against ground-truth labels for regional dialect distress signals.\n")

    mapper = MockSentimentMapper(embedding_dim=768, seed=42)

    # Ground-truth labeled client communications
    test_samples = [
        ("My crops failed this season, I cannot repay", "critical_risk"),
        ("Had a medical emergency last week", "high_risk"),
        ("The floods destroyed my shop", "critical_risk"),
        ("Lost my job, struggling to find work", "high_risk"),
        ("Business is growing well, paying early", "low_risk"),
        ("Good harvest this year, on time payment", "low_risk"),
        ("New job, income has increased significantly", "low_risk"),
        ("Making regular savings, loan on track", "low_risk"),
        ("Family illness drained savings", "high_risk"),
        ("Drought has affected my farming output", "critical_risk"),
        ("Received a promotion at work", "low_risk"),
        ("Some difficulty with payments this month", "moderate_risk"),
        ("Seasonal hardship but managing", "moderate_risk"),
        ("Debt is becoming overwhelming", "high_risk"),
        ("Business venture is profitable now", "low_risk"),
        ("Late payment due to unexpected expenses", "moderate_risk"),
        ("Crop failure and family illness together", "critical_risk"),
        ("Steady income from new employment", "low_risk"),
        ("Struggling with debt spiral", "high_risk"),
        ("Good savings buffer, no concerns", "low_risk"),
    ]

    risk_to_label = {"low_risk": 0, "moderate_risk": 1, "high_risk": 2, "critical_risk": 3}
    y_true = []
    y_pred = []

    for text, true_label in test_samples:
        tags = mapper.extract_risk_tags(text)
        distress = tags["distress_score"]

        # Map distress score to risk tier
        if distress >= 0.8:
            pred_label = 3  # critical
        elif distress >= 0.6:
            pred_label = 2  # high
        elif distress >= 0.4:
            pred_label = 1  # moderate
        else:
            pred_label = 0  # low

        y_true.append(risk_to_label[true_label])
        y_pred.append(pred_label)

    nlp_metrics = compute_nlp_metrics(np.array(y_true), np.array(y_pred))

    print(f"  {'Metric':<30} {'Value':>10}")
    print(f"  {'─' * 42}")
    print(f"  {'Macro F1-Score':<30} {nlp_metrics['macro_f1']:>10.4f}")
    print(f"  {'Macro Recall':<30} {nlp_metrics['macro_recall']:>10.4f}")
    print()

    label_names = {0: "low_risk", 1: "moderate_risk", 2: "high_risk", 3: "critical_risk"}
    print(f"  Per-Class Breakdown:")
    print(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'─' * 52}")
    for label_id in sorted(nlp_metrics["per_class"].keys(), key=int):
        cls = nlp_metrics["per_class"][label_id]
        name = label_names.get(int(label_id), label_id)
        print(f"  {name:<20} {cls['precision']:>10.4f} {cls['recall']:>10.4f} {cls['f1']:>10.4f}")

    f1_pass = nlp_metrics["macro_f1"] >= 0.5
    recall_pass = nlp_metrics["macro_recall"] >= 0.5
    print(f"\n  Macro F1 ≥ 0.50:   {'✅ PASS' if f1_pass else '❌ FAIL'}")
    print(f"  Macro Recall ≥ 0.50: {'✅ PASS' if recall_pass else '❌ FAIL'}")

    # ===================================================================
    # DIMENSION 2: ENVIRONMENT FIDELITY (Calibration)
    # ===================================================================
    print_header("DIMENSION 2: ENVIRONMENT FIDELITY (Calibration)")
    print("  K-S Test: synthetic client distributions vs. historical Mifos X data")
    print("  MAE: simulated transition outcomes vs. historical step-outcomes\n")

    # Load and calibrate from sample data
    calibrator = MifosCalibrator()
    sample_path = Path("data/sample_data/mifos_loans.csv")
    df = calibrator.load_csv(sample_path)
    fitted = calibrator.fit_distributions(df)

    print(f"  Loaded {len(df)} historical records from Mifos X sample data\n")

    # Generate synthetic data from fitted distributions
    rng = np.random.default_rng(123)
    n_synthetic = 10000

    features_to_test = [
        "credit_score", "dti_ratio", "income_stability",
        "repayment_history", "loan_amount_norm", "unemployment_indicator",
    ]

    print(f"  K-S Test Results (H0: same distribution, α=0.05):")
    print(f"  {'Feature':<25} {'K-S Stat':>10} {'p-value':>10} {'Result':>10}")
    print(f"  {'─' * 57}")

    ks_all_passed = True
    for feature in features_to_test:
        historical = df[feature].dropna().values
        historical = np.clip(historical, 0.001, 0.999)

        p = fitted.get(feature, {})
        if "alpha" in p:
            synthetic = rng.beta(p["alpha"], p["beta"], size=n_synthetic)
        else:
            synthetic = np.clip(rng.normal(p.get("mean", 0.5), p.get("std", 0.1), size=n_synthetic), 0, 1)

        ks_result = ks_test_fidelity(historical, synthetic, feature)
        status = "✅ PASS" if ks_result["passed"] else "❌ FAIL"
        if not ks_result["passed"]:
            ks_all_passed = False

        print(f"  {feature:<25} {ks_result['ks_statistic']:>10.4f} {ks_result['p_value']:>10.4f} {status:>10}")

    print(f"\n  Overall K-S Fidelity: {'✅ ALL PASSED' if ks_all_passed else '❌ SOME FAILED'}")

    # MAE: Simulate transitions and compare with historical outcomes
    print(f"\n  MAE — Transition Fidelity:")
    print(f"  Comparing simulated default outcomes against historical default rates\n")

    env = gym.make("MicroLoan-v0", config={
        "use_nlp_features": False, "sentiment_dim": 0
    })

    # Historical default rate from sample data
    hist_defaults = df["didDefault"].values if "didDefault" in df.columns else np.zeros(len(df))
    hist_rates = df["interestRate"].values if "interestRate" in df.columns else np.full(len(df), 0.15)

    # Simulate same number of transitions
    sim_defaults = []
    env.reset(seed=42)
    for i in range(min(len(df), 500)):
        action = int(np.clip(hist_rates[i] / 0.30 * 9, 0, 9))  # Map rate to tier
        obs, reward, term, trunc, info = env.step(action)
        sim_defaults.append(1 if info.get("did_default", False) else 0)
        if term or trunc:
            env.reset()

    mae_result = mae_transition_fidelity(
        hist_defaults[:len(sim_defaults)],
        np.array(sim_defaults),
    )

    print(f"  {'Metric':<35} {'Value':>10}")
    print(f"  {'─' * 47}")
    print(f"  {'MAE (Default Outcomes)':<35} {mae_result['mae']:>10.4f}")
    print(f"  {'Historical Default Rate':<35} {np.mean(hist_defaults):>10.4f}")
    print(f"  {'Simulated Default Rate':<35} {np.mean(sim_defaults):>10.4f}")
    print(f"  {'Bias (Sim - Hist)':<35} {mae_result.get('bias', 0):>10.4f}")
    print(f"  {'N Samples':<35} {mae_result['n_samples']:>10d}")
    env.close()

    # ===================================================================
    # DIMENSION 3: RL POLICY PERFORMANCE (Financial)
    # ===================================================================
    print_header("DIMENSION 3: RL POLICY PERFORMANCE (Financial)")
    print("  Evaluating trained PPO agent over 100 episodes.\n")

    model = PPO.load("checkpoints/final_model")
    eval_env = gym.make("MicroLoan-v0", config={"use_nlp_features": True, "sentiment_dim": 768})

    all_rewards = []
    all_rates = []
    all_defaults_list = []
    all_demo_groups = []
    all_lang_groups = []
    ep_yields = []
    ep_default_rates = []

    n_episodes = 100
    for ep in range(n_episodes):
        obs, info = eval_env.reset()
        ep_r, ep_rates_ep, ep_def, ep_steps, done = 0, [], 0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = eval_env.step(action)
            done = term or trunc
            ep_r += reward
            ep_steps += 1
            if "offered_rate" in info:
                ep_rates_ep.append(info["offered_rate"])
                all_rates.append(info["offered_rate"])
            if info.get("did_default", False):
                ep_def += 1
                all_defaults_list.append(1)
            else:
                all_defaults_list.append(0)
            all_demo_groups.append(info.get("demographic_group", 0))
            all_lang_groups.append(info.get("language_group", 0))
        all_rewards.append(ep_r)
        ep_yields.append(np.mean(ep_rates_ep) if ep_rates_ep else 0)
        ep_default_rates.append(ep_def / max(ep_steps, 1))

    avg_yield = float(np.mean(ep_yields))
    avg_default = float(np.mean(ep_default_rates))

    print(f"  {'Metric':<35} {'Value':>12}")
    print(f"  {'─' * 49}")
    print(f"  {'Avg Episode Reward':<35} {np.mean(all_rewards):>12.3f}")
    print(f"  {'Std Episode Reward':<35} {np.std(all_rewards):>12.3f}")
    print(f"  {'Avg Yield (Interest Rate)':<35} {avg_yield:>12.4f}  ({avg_yield*100:.2f}%)")
    print(f"  {'Avg Default Rate':<35} {avg_default:>12.4f}  ({avg_default*100:.2f}%)")
    print(f"  {'Avg Offered Rate':<35} {np.mean(all_rates):>12.4f}")
    print(f"  {'N Episodes':<35} {n_episodes:>12d}")

    # Pareto Dominance
    baseline_yield = 0.12
    baseline_default = 0.15
    par = pareto_dominance(baseline_yield, baseline_default, avg_yield, avg_default)

    print(f"\n  Pareto Dominance vs. Historical Baseline:")
    print(f"  {'─' * 49}")
    print(f"  {'Baseline Yield':<35} {baseline_yield:>12.4f}")
    print(f"  {'Baseline Default Rate':<35} {baseline_default:>12.4f}")
    print(f"  {'Policy Yield':<35} {avg_yield:>12.4f}")
    print(f"  {'Policy Default Rate':<35} {avg_default:>12.4f}")
    print(f"  {'Yield Δ':<35} {par['yield_improvement']:>+12.4f}")
    print(f"  {'Default Δ (lower=better)':<35} {par['default_improvement']:>+12.4f}")
    dom_status = "✅ YES — Policy Pareto-dominates baseline" if par["pareto_dominates"] else "❌ NO — Welfare-first strategy"
    print(f"  {'Pareto Dominates?':<35} {dom_status}")

    # Policy Stability
    stab = policy_stability(np.array(all_rates), window_size=100)

    print(f"\n  Policy Stability:")
    print(f"  {'─' * 49}")
    print(f"  {'Overall Action Variance':<35} {stab['overall_variance']:>12.6f}")
    print(f"  {'Converging':<35} {'✅ Yes' if stab.get('converging', True) else '❌ No':>12}")
    print(f"  {'Stable (final window var < 0.01)':<35} {'✅ Yes' if stab['stable'] else '❌ No':>12}")
    if stab.get("rolling_variances"):
        print(f"  {'Final Window Variance':<35} {stab['rolling_variances'][-1]:>12.6f}")
        print(f"  {'First Window Variance':<35} {stab['rolling_variances'][0]:>12.6f}")

    eval_env.close()

    # ===================================================================
    # DIMENSION 4: FAIRNESS METRICS (Ethical)
    # ===================================================================
    print_header("DIMENSION 4: FAIRNESS METRICS (Ethical)")
    print("  Measuring SPD and DI Ratio across demographic and language groups.\n")

    rates_arr = np.array(all_rates)
    demo_arr = np.array(all_demo_groups)
    lang_arr = np.array(all_lang_groups)

    report = compute_fairness_report(rates_arr, demo_arr, lang_arr)
    demo = report["demographic"]
    lang = report.get("language", {})

    # --- Demographic ---
    print(f"  DEMOGRAPHIC GROUP (Majority vs. Minority)")
    print(f"  {'─' * 60}")
    print(f"  Formula:")
    print(f"    SPD = P(Ŷ=1|A=minority) - P(Ŷ=1|A=majority)")
    print(f"    DI  = P(Ŷ=1|A=minority) / P(Ŷ=1|A=majority)")
    print()
    print(f"  {'Metric':<40} {'Value':>10} {'Status':>12}")
    print(f"  {'─' * 64}")
    spd_val = demo.get("spd", 0)
    di_val = demo.get("di_ratio", 1)
    print(f"  {'Statistical Parity Difference (SPD)':<40} {spd_val:>+10.4f} {'✅ PASS' if abs(spd_val) <= 0.1 else '❌ FAIL':>12}")
    print(f"  {'Disparate Impact (DI) Ratio':<40} {di_val:>10.4f} {'✅ PASS' if 0.8 <= di_val <= 1.25 else '❌ FAIL':>12}")
    print(f"  {'Avg Rate — Majority':<40} {demo.get('avg_rate_majority', 0):>10.4f}")
    print(f"  {'Avg Rate — Minority':<40} {demo.get('avg_rate_minority', 0):>10.4f}")
    print(f"  {'N Majority':<40} {demo.get('n_majority', 0):>10d}")
    print(f"  {'N Minority':<40} {demo.get('n_minority', 0):>10d}")
    print(f"  {'DI in [0.80, 1.25]?':<40} {'✅ COMPLIANT' if demo.get('di_compliant') else '❌ VIOLATION':>12}")
    print(f"  {'|SPD| ≤ 0.10?':<40} {'✅ COMPLIANT' if demo.get('spd_compliant') else '❌ VIOLATION':>12}")

    # --- Language ---
    if lang:
        print(f"\n  LANGUAGE GROUP (High-Resource vs. Low-Resource)")
        print(f"  {'─' * 64}")
        lang_spd = lang.get("spd", 0)
        lang_di = lang.get("di_ratio", 1)
        print(f"  {'Statistical Parity Difference (SPD)':<40} {lang_spd:>+10.4f} {'✅ PASS' if abs(lang_spd) <= 0.1 else '❌ FAIL':>12}")
        print(f"  {'Disparate Impact (DI) Ratio':<40} {lang_di:>10.4f} {'✅ PASS' if 0.8 <= lang_di <= 1.25 else '❌ FAIL':>12}")
        print(f"  {'Avg Rate — HRL (High-Resource Lang)':<40} {lang.get('avg_rate_hrl', 0):>10.4f}")
        print(f"  {'Avg Rate — LRL (Low-Resource Lang)':<40} {lang.get('avg_rate_lrl', 0):>10.4f}")
        print(f"  {'DI in [0.80, 1.25]?':<40} {'✅ COMPLIANT' if lang.get('di_compliant') else '❌ VIOLATION':>12}")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    # Pre-compute metric strings to avoid f-string nesting issues
    f1_str = f"F1={nlp_metrics['macro_f1']:.3f}"
    var_str = f"var={stab['overall_variance']:.4f}"
    di_str = f"DI={di_val:.3f}"
    nlp_status = "✅ PASS" if f1_pass else "❌ FAIL"
    ks_status = "✅ PASS" if ks_all_passed else "❌ FAIL"
    stab_status = "✅ PASS" if stab["stable"] else "❌ FAIL"
    fair_status = "✅ PASS" if demo.get("di_compliant") else "❌ FAIL"

    print(f"\n{HEADER}")
    print(f"  SUMMARY — 4-DIMENSION EVALUATION")
    print(HEADER)
    print(f"\n  {'Dimension':<35} {'Key Metric':>15} {'Status':>12}")
    print(f"  {'─' * 64}")
    print(f"  {'1. NLP Subsystem':<35} {f1_str:>15} {nlp_status:>12}")
    print(f"  {'2. Env Fidelity (K-S)':<35} {'p>0.05':>15} {ks_status:>12}")
    print(f"  {'3. RL Policy (Stability)':<35} {var_str:>15} {stab_status:>12}")
    print(f"  {'4. Fairness (DI Ratio)':<35} {di_str:>15} {fair_status:>12}")
    print(f"\n{HEADER}")
    print(f"  EVALUATION COMPLETE")
    print(f"{HEADER}\n")


if __name__ == "__main__":
    main()
