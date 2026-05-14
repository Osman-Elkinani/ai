# 🏋️ AI for Personalized Health & Fitness Recommendations

An AI system that tracks user fitness data and provides personalized workout and nutrition recommendations using **Reinforcement Learning** and **Informed Search** algorithms.

## 🎯 Project Overview

This system models a user's fitness journey as a **Markov Decision Process (MDP)** and uses three AI agents to find optimal health recommendations:

| Agent | Algorithm | Lecture | Description |
|-------|-----------|---------|-------------|
| **Q-Learning** | Tabular Q-Learning | RL I (L9) | Model-free learning of optimal workout/nutrition policies |
| **Approximate Q-Learning** | Feature-based Q-Learning | RL II (L10) | Generalized learning using feature extraction |
| **A\* Search** | Informed Heuristic Search | Informed Search (L3) | Optimal plan finding using admissible heuristics |

## 🧠 AI Concepts Demonstrated

- **MDP Formulation** (Lectures 7-8): States, Actions, Transitions, Rewards, Discount Factor
- **A\* Search** (Lecture 3): Optimal planning with admissible heuristic h(n)
- **Q-Learning** (Lecture 9): Model-free, off-policy TD learning
- **Approximate Q-Learning** (Lecture 10): Feature-based state representation
- **Epsilon-Greedy Exploration** (Lecture 10): Exploration vs Exploitation tradeoff
- **Graph Search** (Lecture 3): Closed set to avoid revisiting states

## 📊 MDP Formulation

- **State Space** (~43,750 states): (fitness, energy, weight, muscle, fatigue, day)
- **Action Space** (28 actions): 7 workout types × 4 nutrition plans
- **Transitions**: Stochastic state changes with ±20% noise
- **Rewards**: Goal-dependent (weight loss, muscle gain, general fitness, endurance)
- **Discount Factor**: γ = 0.95

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the web dashboard
python run.py

# 3. Open your browser to http://127.0.0.1:5000
```

### CLI Demo (no web browser needed)
```bash
python run.py --cli
```

## 📁 Project Structure

```
Project/
├── run.py                          # Main entry point
├── config.py                       # All configuration & hyperparameters
├── requirements.txt                # Python dependencies
├── environment/
│   ├── fitness_env.py              # Custom Gym-style MDP environment
│   └── user_profile.py             # User profiles and goal definitions
├── agents/
│   ├── q_learning_agent.py         # Tabular Q-Learning (Lecture 9)
│   ├── approx_q_agent.py           # Approximate Q-Learning (Lecture 10)
│   └── search_agent.py             # A* Search (Lecture 3)
├── data/
│   ├── workouts.py                 # Workout database and effects
│   └── nutrition.py                # Nutrition plan database
└── web/
    ├── app.py                      # Flask API server
    ├── templates/dashboard.html    # Dashboard UI
    └── static/
        ├── css/style.css           # Premium dark theme
        └── js/dashboard.js         # Interactive charts & controls
```

## 🖥️ Dashboard Features

1. **User Profile Configuration** — Select from preset profiles or customize
2. **Agent Training** — Train Q-Learning and Approximate Q-Learning agents
3. **A\* Search** — Find optimal multi-day workout plans
4. **Live Simulation** — Watch the agent make daily recommendations
5. **Learning Curve Chart** — Visualize agent improvement over episodes
6. **Fitness Progress Chart** — Track simulated health metrics
7. **Performance Comparison** — Compare RL vs Search approaches

## 🔧 Configuration

Key hyperparameters can be adjusted in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LEARNING_RATE` | 0.1 | Q-value update step size (α) |
| `DISCOUNT_FACTOR` | 0.95 | Future reward importance (γ) |
| `EPSILON_START` | 1.0 | Initial exploration rate (ε) |
| `EPSILON_DECAY` | 0.995 | Exploration decay per episode |
| `NUM_EPISODES` | 500 | Default training episodes |
| `MAX_STEPS_PER_EPISODE` | 30 | Days per simulation episode |

## 👥 Team

University AI Course — Semester Project

## 📝 License

This project is for educational purposes only.
