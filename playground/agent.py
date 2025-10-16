import numpy as np

class QLearningAgent:
    """
    Einfacher tabellarischer Q-Learning-Agent.
    - epsilon-greedy Aktionswahl
    - klassisches Q-Update
    - optionaler Epsilon-Decay
    """
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.999
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

    def select_action(self, state: int) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, s: int, a: int, r: float, s_next: int, done: bool):
        best_next = 0.0 if done else float(np.max(self.Q[s_next]))
        td_target = r + self.gamma * best_next
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def greedy_action(self, state: int) -> int:
        return int(np.argmax(self.Q[state]))
