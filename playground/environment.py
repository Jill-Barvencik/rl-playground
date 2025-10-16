from __future__ import annotations
import numpy as np
import gymnasium as gym
from typing import Optional, Tuple, List


# -------- Gemeinsame Adapter-Schnittstelle (diskrete Zustände) --------

class TabularEnvAdapter:
    """
    Vereinheitlichte Schnittstelle für tabulares RL:
    - n_states, n_actions
    - reset() -> state_id (int)
    - step(action_id) -> (state_id, reward, done, info)
    """

    n_states: int
    n_actions: int

    def reset(self) -> int:
        raise NotImplementedError

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        raise NotImplementedError


# -------- FrozenLake: bereits diskret --------

class FrozenLakeAdapter(TabularEnvAdapter):
    def __init__(self, is_slippery: bool = False, map_name: str = "4x4"):
        self.env = gym.make("FrozenLake-v1", is_slippery=is_slippery, map_name=map_name)
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self._state = 0

    def reset(self) -> int:
        s, _ = self.env.reset()
        self._state = int(s)
        return self._state

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        s_next, r, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        self._state = int(s_next)
        return int(s_next), float(r), done, info


# -------- MountainCar: Diskretisierung (Position, Velocity) --------

class MountainCarAdapter(TabularEnvAdapter):
    """
    Diskretisiert (position, velocity) in Bins.
    Aktionen (0=left,1=neutral,2=right) sind bereits diskret.
    """
    def __init__(self, bins_pos: int = 18, bins_vel: int = 14, seed: Optional[int] = 42):
        self.env = gym.make("MountainCar-v0")
        if seed is not None:
            self.env.reset(seed=seed)
            np.random.seed(seed)
        self.bins_pos = int(bins_pos)
        self.bins_vel = int(bins_vel)

        # Beobachtungsgrenzen aus Env
        self.pos_low, self.pos_high = self.env.observation_space.low[0], self.env.observation_space.high[0]
        self.vel_low, self.vel_high = self.env.observation_space.low[1], self.env.observation_space.high[1]

        self.pos_bins = np.linspace(self.pos_low, self.pos_high, self.bins_pos + 1)
        self.vel_bins = np.linspace(self.vel_low, self.vel_high, self.bins_vel + 1)

        self.n_actions = self.env.action_space.n  # 3
        self.n_states = self.bins_pos * self.bins_vel
        self._state = 0

    def _obs_to_state(self, obs: np.ndarray) -> int:
        pos, vel = obs[0], obs[1]
        i = np.clip(np.digitize(pos, self.pos_bins) - 1, 0, self.bins_pos - 1)
        j = np.clip(np.digitize(vel, self.vel_bins) - 1, 0, self.bins_vel - 1)
        return int(i * self.bins_vel + j)

    def reset(self) -> int:
        obs, _ = self.env.reset()
        s = self._obs_to_state(obs)
        self._state = s
        return s

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        obs_next, r, terminated, truncated, info = self.env.step(action)
        s_next = self._obs_to_state(obs_next)
        done = bool(terminated or truncated)
        self._state = s_next
        return s_next, float(r), done, info


# -------- Energy Storage: simple, diskrete Custom-Umgebung --------

class EnergyStorageEnv:
    """
    Simpler, didaktischer Speicher:
    - Diskrete Zeit t = 0..H-1 (H = horizon)
    - Diskrete Ladestände soc in [0, capacity_levels-1]
    - Aktionen: 0=discharge, 1=hold, 2=charge (jeweils +/-1 Stufe)
    - Preisfolge p_t (synthetisch): Basismuster + Rauschen
    - Reward = -(cost) = -(price_t * net_grid_power)
      Annahme: discharge verkauft (Erlös), charge kauft (Kosten)
    Keine Wirkungsgrade/Verluste (didaktisch).
    """

    def __init__(
        self,
        horizon: int = 48,
        capacity_levels: int = 6,
        max_step: int = 1,           # Stufenänderung pro Schritt
        price_volatility: float = 0.5,
        seed: Optional[int] = 42,
    ):
        self.horizon = int(horizon)
        self.capacity_levels = int(capacity_levels)
        self.max_step = int(max_step)
        self.price_volatility = float(price_volatility)
        self.rng = np.random.default_rng(seed)
        self.reset_prices()

        self.t = 0
        self.soc = capacity_levels // 2  # Start in der Mitte

    def reset_prices(self):
        t = np.arange(self.horizon)
        base = 0.5 + 0.5 * np.sin(2 * np.pi * t / max(1, self.horizon // 2))  # Tagesmuster
        noise = self.rng.normal(0, self.price_volatility, size=self.horizon) * 0.1
        self.prices = np.clip(base + noise, 0.0, None)

    @property
    def n_states(self) -> int:
        # Zustand = (t, soc) → flach kodiert
        return self.horizon * self.capacity_levels

    @property
    def n_actions(self) -> int:
        # discharge, hold, charge
        return 3

    def _encode(self, t: int, soc: int) -> int:
        return t * self.capacity_levels + soc

    def reset(self) -> int:
        self.t = 0
        self.soc = self.capacity_levels // 2
        return self._encode(self.t, self.soc)

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        assert action in (0, 1, 2)
        # Aktion in Δsoc übersetzen
        delta = {0: -self.max_step, 1: 0, 2: +self.max_step}[action]
        next_soc = int(np.clip(self.soc + delta, 0, self.capacity_levels - 1))

        price = float(self.prices[self.t])
        # Net grid power (einfach): charge -> +1 (kaufen, Kosten), discharge -> -1 (verkaufen, Erlös)
        net_grid = 0
        if next_soc > self.soc:
            net_grid = +1
        elif next_soc < self.soc:
            net_grid = -1

        cost = price * net_grid
        reward = -cost  # Kosten minimieren → Reward = -(Kosten)
        self.soc = next_soc
        self.t += 1
        done = self.t >= self.horizon
        info = {"price": price, "soc": self.soc, "t": self.t}
        return self._encode(min(self.t, self.horizon - 1), self.soc), float(reward), bool(done), info


class EnergyStorageAdapter(TabularEnvAdapter):
    def __init__(self, horizon: int = 48, capacity_levels: int = 6, price_volatility: float = 0.5, seed: Optional[int] = 42):
        self.env = EnergyStorageEnv(horizon=horizon, capacity_levels=capacity_levels, price_volatility=price_volatility, seed=seed)
        self.n_states = self.env.n_states
        self.n_actions = self.env.n_actions
        self._state = 0

    def reset(self) -> int:
        self._state = self.env.reset()
        return self._state

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        s_next, r, done, info = self.env.step(action)
        self._state = s_next
        return s_next, r, done, info


# -------- Fabrikfunktion für UI --------

def make_adapter(
    kind: str,
    *,
    is_slippery: bool = False,
    map_name: str = "4x4",
    bins_pos: int = 18,
    bins_vel: int = 14,
    storage_horizon: int = 48,
    storage_levels: int = 6,
    storage_volatility: float = 0.5,
    seed: Optional[int] = 42,
) -> TabularEnvAdapter:
    kind = kind.lower()
    if kind == "frozenlake":
        return FrozenLakeAdapter(is_slippery=is_slippery, map_name=map_name)
    elif kind == "mountaincar":
        return MountainCarAdapter(bins_pos=bins_pos, bins_vel=bins_vel, seed=seed)
    elif kind == "energy storage":
        return EnergyStorageAdapter(horizon=storage_horizon, capacity_levels=storage_levels, price_volatility=storage_volatility, seed=seed)
    else:
        raise ValueError(f"Unknown environment kind: {kind}")
