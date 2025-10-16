# playground/deep.py
from __future__ import annotations
from typing import Tuple, Optional, List
import numpy as np
import gymnasium as gym

def _make_env(env_id: str, seed: Optional[int] = 42):
    env = gym.make(env_id)
    if seed is not None:
        env.reset(seed=seed)
    return env

def train_dqn(
    env_id: str,
    total_timesteps: int = 20_000,
    seed: int = 42,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    target_update_interval: int = 1_000,
) -> Tuple["DQN", List[float]]:
    """
    Trainiert ein DQN (SB3) auf CartPole-v1 oder MountainCar-v0 und
    gibt (modell, episodische_rewards) zurück.
    """
    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
    except Exception as e:
        raise RuntimeError(
            "stable-baselines3/torch not installed. Please `pip install stable-baselines3 torch`."
        ) from e

    class RewardCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_rewards: List[float] = []

        def _on_step(self) -> bool:
            # Monitor verpackt Infos – episodische Rewards in infos mit 'episode'
            infos = self.locals.get("infos", [])
            for info in infos:
                ep = info.get("episode")
                if ep is not None and "r" in ep:
                    self.episode_rewards.append(float(ep["r"]))
            return True

    # Monitor sammelt episodische Infos automatisch
    env = _make_env(env_id, seed)
    env = Monitor(env)

    policy = "MlpPolicy"
    model = DQN(
        policy,
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        target_update_interval=target_update_interval,
        verbose=0,
        seed=seed,
        tensorboard_log=None,
    )

    cb = RewardCallback()
    model.learn(total_timesteps=total_timesteps, callback=cb, progress_bar=False)
    return model, cb.episode_rewards
