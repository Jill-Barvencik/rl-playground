import numpy as np
from playground.agent import QLearningAgent

def test_q_update():
    agent = QLearningAgent(n_states=4, n_actions=2, alpha=1.0, gamma=0.0, epsilon=0.0)
    # Terminaler Schritt: next state's best Q nicht mehr relevant
    agent.update(s=0, a=1, r=1.0, s_next=3, done=True)
    assert np.isclose(agent.Q[0, 1], 1.0)
