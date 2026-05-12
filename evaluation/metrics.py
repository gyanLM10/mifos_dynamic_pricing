"""
Evaluation Metrics — K-S test, MAE, Pareto dominance, policy stability,
NLP F1/Recall, and a comprehensive evaluation runner.
"""
from __future__ import annotations
import logging
import numpy as np
from scipy import stats
from typing import Any

logger = logging.getLogger(__name__)

def compute_nlp_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    unique = np.unique(np.concatenate([y_true, y_pred]))
    per_class = {}
    for label in unique:
        tp = ((y_pred == label) & (y_true == label)).sum()
        fp = ((y_pred == label) & (y_true != label)).sum()
        fn = ((y_pred != label) & (y_true == label)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[str(label)] = {"precision": float(p), "recall": float(r), "f1": float(f1)}
    return {"macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
            "macro_recall": float(np.mean([v["recall"] for v in per_class.values()])),
            "per_class": per_class}

def ks_test_fidelity(historical, synthetic, feature_name="unknown"):
    ks_stat, p_value = stats.ks_2samp(np.asarray(historical).flatten(), np.asarray(synthetic).flatten())
    return {"feature": feature_name, "ks_statistic": float(ks_stat), "p_value": float(p_value), "passed": p_value > 0.05}

def mae_transition_fidelity(historical_outcomes, simulated_outcomes):
    h, s = np.asarray(historical_outcomes).flatten(), np.asarray(simulated_outcomes).flatten()
    n = min(len(h), len(s))
    return {"mae": float(np.mean(np.abs(h[:n] - s[:n]))), "n_samples": n}

def pareto_dominance(baseline_yield, baseline_default, policy_yield, policy_default):
    yi = policy_yield - baseline_yield
    di = baseline_default - policy_default
    return {"pareto_dominates": (yi >= 0 and di > 0) or (yi > 0 and di >= 0),
            "yield_improvement": float(yi), "default_improvement": float(di)}

def policy_stability(action_history, window_size=100):
    a = np.asarray(action_history).flatten()
    if len(a) < window_size:
        return {"overall_variance": float(np.var(a)), "stable": True}
    rolling = [float(np.var(a[i:i+window_size])) for i in range(0, len(a)-window_size+1, window_size//2)]
    return {"overall_variance": float(np.var(a)), "rolling_variances": rolling,
            "converging": np.mean(rolling[len(rolling)//2:]) < np.mean(rolling[:len(rolling)//2]) if len(rolling) >= 2 else True,
            "stable": rolling[-1] < 0.01 if rolling else True}

def run_full_evaluation(env, model, n_episodes=100, baseline_yield=0.12, baseline_default_rate=0.15):
    from safety.fairness_metrics import compute_fairness_report
    all_rewards, all_rates, all_demo, all_lang = [], [], [], []
    ep_yields, ep_defaults = [], []
    for _ in range(n_episodes):
        obs, info = env.reset()
        ep_r, ep_rates, ep_def, ep_steps, done = 0, [], 0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            ep_r += reward; ep_steps += 1
            if "offered_rate" in info:
                ep_rates.append(info["offered_rate"]); all_rates.append(info["offered_rate"])
            if info.get("did_default", False): ep_def += 1
            all_demo.append(info.get("demographic_group", 0))
            all_lang.append(info.get("language_group", 0))
        all_rewards.append(ep_r)
        ep_yields.append(np.mean(ep_rates) if ep_rates else 0)
        ep_defaults.append(ep_def / max(ep_steps, 1))
    avg_y, avg_d = float(np.mean(ep_yields)), float(np.mean(ep_defaults))
    return {
        "policy": {"avg_reward": float(np.mean(all_rewards)), "avg_yield": avg_y, "avg_default_rate": avg_d},
        "pareto": pareto_dominance(baseline_yield, baseline_default_rate, avg_y, avg_d),
        "stability": policy_stability(np.array(all_rates)),
        "fairness": compute_fairness_report(np.array(all_rates), np.array(all_demo), np.array(all_lang)),
    }
