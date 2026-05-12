# Multimodal RL for Inclusive Micro-Lending

A simulated environment for testing dynamic pricing strategies for micro-loans using
Multimodal Reinforcement Learning. The system fuses **tabular financial data** with
**cross-lingual NLP sentiment** (XLM-RoBERTa) to learn personalized interest-rate
policies that balance institutional sustainability, client welfare, and ethical inclusion.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Supervisor Agent (LangGraph)                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐    │
│  │ MCP Tool Call │──▶│  NLP Server  │──▶│ Sentiment Vector │    │
│  │ (Rust Edge)   │   │  (XLM-R)     │   │ Injection        │    │
│  └──────────────┘   └──────────────┘   └──────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Gymnasium Environment (MicroLoanEnv)               │
│  ┌──────────────────┐  ┌───────────────┐  ┌────────────────┐   │
│  │ Tabular Features │  │ NLP Embedding │  │ Econ Indicators│   │
│  │ (DTI, history)   │  │ (768-d vector)│  │ (GDP, unemp.)  │   │
│  └────────┬─────────┘  └──────┬────────┘  └───────┬────────┘   │
│           └────────────────────┼────────────────────┘            │
│                                ▼                                │
│               Composite State Space (S)                         │
│                                │                                │
│                     ┌──────────┴──────────┐                     │
│                     │   RL Policy (PPO)   │                     │
│                     │  Late Fusion Network│                     │
│                     └──────────┬──────────┘                     │
│                                ▼                                │
│                     Interest Rate Action (A)                    │
│                                │                                │
│              ┌─────────────────┼─────────────────┐              │
│              ▼                 ▼                  ▼              │
│     CPO Guardrails     Reward R(s,a)      Fairness Check        │
│     (rate caps)        (multi-obj)        (SPD, DI ratio)       │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
dynamic-pricing-rl/
├── env/                    # Gymnasium environment
│   ├── microloan_env.py    # Core MDP implementation
│   └── client_generator.py # Synthetic client profiles
├── data/                   # Mifos X data calibration
│   ├── mifos_calibrator.py
│   └── sample_data/
├── nlp/                    # NLP pipeline
│   ├── sentiment_mapper.py # XLM-RoBERTa encoder
│   └── label_studio_bridge.py
├── agents/                 # RL agents
│   ├── multimodal_policy.py# Late-fusion policy network
│   ├── train.py            # Training orchestrator
│   └── curriculum.py       # Weight annealing
├── safety/                 # Ethical guardrails
│   ├── guardrails.py       # CPO + rate caps
│   └── fairness_metrics.py # SPD, DI calculations
├── evaluation/             # Metrics & visualization
│   ├── metrics.py          # K-S, MAE, Pareto, F1
│   └── visualize.py        # Plotting utilities
└── tests/                  # Unit tests
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run environment validation
python -m pytest tests/ -v

# Train an agent (PPO with curriculum learning)
python -m agents.train --algo ppo --episodes 5000

# Evaluate fairness
python -m evaluation.metrics --policy checkpoints/best_model.zip
```

## Reward Function

$$R = w_1(\text{Interest Earned}) - w_2(\text{Default Risk}) + w_3(\text{Client Utility}) - \text{Opportunity Cost}$$

| Weight | Suggested Range | Purpose |
|--------|----------------|---------|
| $w_1$  | 1.0            | Institutional sustainability |
| $w_2$  | 1.2 – 1.5      | Default risk aversion |
| $w_3$  | 0.5 – 0.8      | Client welfare & accessibility |

## Fairness Targets

- **Disparate Impact (DI) Ratio**: $0.8 \leq DI \leq 1.25$
- **Statistical Parity Difference (SPD)**: $|SPD| \leq 0.1$

## References

- [XLM-RoBERTa (Conneau et al., 2019)](https://arxiv.org/abs/1911.02116)
- [UNESCO AI & Language Preservation](https://en.unesco.org/artificial-intelligence/language-preservation)
- [Mifos X Platform](https://mifos.org/)
- [Label Studio](https://labelstud.io/)
