# Reinforcement Learning Playground

[![CI](https://github.com/Jill-Barvencik/rl-playground/actions/workflows/ci.yml/badge.svg)](https://github.com/Jill-Barvencik/rl-playground/actions/workflows/ci.yml)

## Overview

The Reinforcement Learning (RL) Playground is an interactive environment designed for educational and research purposes. Its primary objective is to make the core principles of Reinforcement Learning—such as agent–environment interaction, reward mechanisms, and policy optimization—accessible and intuitive through interactive experimentation and visualization.

---

## Objectives

- Provide an interactive and visual interface to explore how RL agents learn over time.  
- Allow users to manipulate learning parameters (learning rate, discount factor, exploration rate, number of episodes) and observe their effects.  
- Demonstrate the interaction between environment dynamics, reward design, and policy behavior.  

---

## Repository Structure

```
├── rl_playground/       # Core RL components
│   ├── agent.py         # Q-Learning agent implementation
│   ├── environment.py   # Environment adapters (FrozenLake, MountainCar, etc.)
│   ├── deep.py          # Deep Q-Network (DQN) implementation
│   ├── ui.py            # Streamlit web interface
│   └── utils.py         # Utility functions
├── tests/               # Unit tests
├── pyproject.toml       # Project configuration and dependencies
├── .python-version      # Python version specification
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended for dependency management)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Jill-Barvencik/rl-playground.git
   cd rl-playground
   ```

2. **Install dependencies with uv:**
   ```bash
   uv sync
   ```

   Or if you prefer pip:
   ```bash
   pip install -e .
   ```

3. **Launch the application:**
   ```bash
   uv run streamlit run rl_playground/ui.py
   ```

   Or with pip:
   ```bash
   streamlit run rl_playground/ui.py
   ```

4. **Open your browser** to `http://localhost:8501`

---

## Development

### Running Tests

```bash
uv run pytest
```

### Development Dependencies

Install with development dependencies:
```bash
uv sync --dev
```

This includes additional tools like `pytest`, `ruff`, and `mypy` for testing and code quality.

---

## Usage

The playground provides an interactive web interface where you can:

1. **Choose algorithms**: Tabular Q-Learning or Deep Q-Network (DQN)
2. **Select environments**: FrozenLake, MountainCar, Energy Storage, CartPole
3. **Adjust parameters**: Learning rate, exploration rate, episodes, etc.
4. **Watch training**: Real-time visualization of agent learning
5. **Analyze results**: View Q-values, rewards, and learned policies

### Supported Environments

- **FrozenLake**: Navigate a grid world avoiding holes
- **MountainCar**: Build momentum to reach a hilltop
- **Energy Storage**: Optimize energy trading decisions
- **CartPole**: Balance a pole using left/right forces (DQN only)

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `uv run pytest`
5. Submit a pull request

---