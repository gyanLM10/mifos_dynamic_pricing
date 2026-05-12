"""
Gymnasium Environment Registration for the MicroLoan Dynamic Pricing Environment.
"""

from gymnasium.envs.registration import register

register(
    id="MicroLoan-v0",
    entry_point="env.microloan_env:MicroLoanEnv",
    max_episode_steps=52,  # 52 weeks = 1 year of weekly decisions
)
