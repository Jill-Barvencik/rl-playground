"""
RL Playground - Interactive Reinforcement Learning Environment

A modern, interactive web-based platform for exploring and learning 
reinforcement learning concepts through hands-on experimentation.
"""

__version__ = "0.1.0"
__author__ = "Jill-Barvencik"

from .agent import QLearningAgent
from .environment import make_adapter

__all__ = ["QLearningAgent", "make_adapter"]