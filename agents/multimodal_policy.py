"""
Multimodal Policy Network — Late-fusion architecture that processes
tabular financial features and NLP sentiment embeddings through separate
branches before fusing in the final layers.

Compatible with Stable Baselines 3 custom feature extractors.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Any


class MultiModalLateFusionExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for the MicroLoanEnv's composite state space.

    Architecture:
        Tabular Branch:  7 → 64 → 128 (ReLU + LayerNorm)
        NLP Branch:      768 → 256 → 128 (ReLU + LayerNorm + Dropout)
        Fusion:          concat(128, 128) → 256 → 128

    The two branches are designed to process fundamentally different signal
    types: structured financial ratios vs. dense semantic embeddings.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        tabular_dim: int = 7,
        nlp_dim: int = 768,
        tabular_hidden: int = 128,
        nlp_hidden: int = 128,
        fusion_hidden: int = 128,
        dropout: float = 0.1,
    ):
        # The output dimension after fusion
        features_dim = fusion_hidden
        super().__init__(observation_space, features_dim=features_dim)

        self.tabular_dim = tabular_dim
        self.nlp_dim = nlp_dim

        # --- Tabular Branch ---
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, tabular_hidden),
            nn.ReLU(),
            nn.LayerNorm(tabular_hidden),
        )

        # --- NLP Branch ---
        if nlp_dim > 0:
            self.nlp_branch = nn.Sequential(
                nn.Linear(nlp_dim, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Dropout(dropout),
                nn.Linear(256, nlp_hidden),
                nn.ReLU(),
                nn.LayerNorm(nlp_hidden),
            )
            fusion_input = tabular_hidden + nlp_hidden
        else:
            self.nlp_branch = None
            fusion_input = tabular_hidden

        # --- Fusion Layers ---
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, fusion_hidden),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Split the composite observation
        tabular = observations[:, :self.tabular_dim]
        tabular_features = self.tabular_branch(tabular)

        if self.nlp_branch is not None and self.nlp_dim > 0:
            nlp = observations[:, self.tabular_dim:]
            nlp_features = self.nlp_branch(nlp)
            fused = torch.cat([tabular_features, nlp_features], dim=-1)
        else:
            fused = tabular_features

        return self.fusion(fused)


class TabularOnlyExtractor(BaseFeaturesExtractor):
    """
    Simpler feature extractor that only processes tabular features.
    Used as a baseline or when NLP features are disabled.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 64,
    ):
        super().__init__(observation_space, features_dim=features_dim)

        obs_dim = int(np.prod(observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


def get_policy_kwargs(
    use_nlp: bool = True,
    tabular_dim: int = 7,
    nlp_dim: int = 768,
) -> dict[str, Any]:
    """
    Build the policy_kwargs dict for Stable Baselines 3.

    Usage:
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, policy_kwargs=get_policy_kwargs())
    """
    if use_nlp:
        return {
            "features_extractor_class": MultiModalLateFusionExtractor,
            "features_extractor_kwargs": {
                "tabular_dim": tabular_dim,
                "nlp_dim": nlp_dim,
            },
            "net_arch": dict(pi=[128, 64], vf=[128, 64]),
        }
    else:
        return {
            "features_extractor_class": TabularOnlyExtractor,
            "features_extractor_kwargs": {"features_dim": 64},
            "net_arch": dict(pi=[64, 32], vf=[64, 32]),
        }
