"""
Q-Learning Agent — Model-Free Reinforcement Learning (Lecture 9: RL I).

Q-Learning is a model-free, off-policy RL algorithm that learns the
optimal action-value function Q*(s, a) directly from experience.

Key concepts from Lecture 9:
  - Q-Value: Q(s, a) = expected total discounted reward from taking
    action a in state s and then following the optimal policy
  - Update Rule: Q(s, a) ← Q(s, a) + α[r + γ·max_a' Q(s', a') - Q(s, a)]
  - This is a form of Temporal Difference (TD) learning
  - No model of the environment is needed (model-free)

Exploration Strategy (Lecture 10: RL II):
  - Epsilon-greedy: with probability ε choose random action,
    otherwise choose the action with highest Q-value
  - ε decays over time: starts high (explore) → ends low (exploit)
"""

import random
import numpy as np
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    LEARNING_RATE, DISCOUNT_FACTOR, EPSILON_START,
    EPSILON_END, EPSILON_DECAY, NUM_EPISODES,
    MAX_STEPS_PER_EPISODE, NUM_ACTIONS
)


class QLearningAgent:
    """
    Tabular Q-Learning agent for the fitness recommendation problem.

    The Q-table maps (state, action) pairs to expected future rewards.
    Through repeated interactions with the environment, the agent
    learns which workout + nutrition combinations are optimal for
    each fitness state.
    """

    def __init__(self, num_actions=NUM_ACTIONS, learning_rate=LEARNING_RATE,
                 discount_factor=DISCOUNT_FACTOR, epsilon_start=EPSILON_START,
                 epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY):
        """
        Initialize the Q-Learning agent.

        Args:
            num_actions: Size of the action space (28)
            learning_rate (α): Step size for Q-value updates
            discount_factor (γ): Importance of future rewards (0.95)
            epsilon_start: Initial exploration rate (1.0)
            epsilon_end: Minimum exploration rate (0.05)
            epsilon_decay: Multiplicative decay per episode (0.995)
        """
        self.num_actions = num_actions
        self.alpha = learning_rate       # α — learning rate
        self.gamma = discount_factor     # γ — discount factor
        self.epsilon = epsilon_start     # ε — exploration rate
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-table with Optimistic Initialization (prevents hole states)
        # Instead of 0.0, unseen (s,a) pairs default to V_max.
        # This forces the agent to explore unvisited states because they
        # appear maximally rewarding — a systematic exploration incentive.
        # Reference: "initialize all Q₀(s,a) to an artificially high V_max"
        self.optimistic_init = 5.0  # V_max: upper bound of expected reward
        self.q_table = defaultdict(lambda: self.optimistic_init)

        # ── Training Metrics ──
        self.episode_rewards = []        # Total reward per episode
        self.episode_steps = []          # Steps per episode
        self.epsilon_history = []        # Epsilon over episodes
        self.states_visited = set()      # Unique states seen
        self.goals_reached = 0           # Episodes that reached the goal
        self.state_visit_counts = defaultdict(int)  # N(s) visit frequency
        self.max_q_delta = 0.0           # Max Q-value change (convergence)

    def get_action(self, state, explore=True):
        """
        Select an action using ε-greedy policy (Lecture 10).

        Exploration vs Exploitation tradeoff:
          - With probability ε: choose RANDOM action (explore)
          - With probability 1-ε: choose BEST action (exploit)

        Args:
            state: Current state tuple
            explore: If False, always pick best action (for evaluation)

        Returns:
            int: Selected action index
        """
        if explore and random.random() < self.epsilon:
            # ── EXPLORATION: random action ──
            return random.randint(0, self.num_actions - 1)
        else:
            # ── EXPLOITATION: pick action with highest Q-value ──
            return self._get_best_action(state)

    def _get_best_action(self, state):
        """
        Return the action with the highest Q-value for the given state.
        Breaks ties randomly.
        """
        q_values = [self.q_table[(state, a)] for a in range(self.num_actions)]
        max_q = max(q_values)

        # Collect all actions with the max Q-value (for tie-breaking)
        best_actions = [a for a in range(self.num_actions) if q_values[a] == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        """
        Update Q-value using the Q-Learning update rule (Lecture 9).

        Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') - Q(s, a)]

        Where:
          - α is the learning rate
          - r is the immediate reward
          - γ is the discount factor
          - max_a' Q(s', a') is the best future value (off-policy)

        Args:
            state: Current state (s)
            action: Action taken (a)
            reward: Reward received (r)
            next_state: Resulting state (s')
            done: Whether the episode ended
        """
        self.states_visited.add(state)
        self.state_visit_counts[state] += 1

        # Current Q-value
        current_q = self.q_table[(state, action)]

        # Maximum Q-value for the next state (the "max" in Q-learning)
        if done:
            max_next_q = 0.0  # Terminal state has no future value
        else:
            max_next_q = max(
                self.q_table[(next_state, a)]
                for a in range(self.num_actions)
            )

        # ── Q-Learning Update Rule ──
        # TD Target = r + γ · max_a' Q(s', a')
        td_target = reward + self.gamma * max_next_q

        # TD Error = TD Target - Current Q
        td_error = td_target - current_q

        # Update Q-value
        new_q = current_q + self.alpha * td_error
        self.q_table[(state, action)] = new_q

        # Track max Q-value delta for convergence monitoring
        q_change = abs(new_q - current_q)
        if q_change > self.max_q_delta:
            self.max_q_delta = q_change

    def decay_epsilon(self):
        """
        Decay ε after each episode to shift from exploration to exploitation.
        ε_new = max(ε_min, ε_old × decay_rate)
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train(self, env, num_episodes=NUM_EPISODES, callback=None):
        """
        Train the agent by interacting with the environment.

        For each episode:
          1. Reset environment to initial state
          2. For each step:
             a. Select action using ε-greedy policy
             b. Execute action, observe (s', r, done)
             c. Update Q-value using the Q-learning rule
          3. Decay ε
          4. Record metrics

        Args:
            env: FitnessEnv instance
            num_episodes: Number of training episodes
            callback: Optional function called after each episode

        Returns:
            dict: Training metrics
        """
        self.episode_rewards = []
        self.episode_steps = []
        self.epsilon_history = []
        self.goals_reached = 0

        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0

            for step in range(MAX_STEPS_PER_EPISODE):
                # 1. Select action (ε-greedy)
                action = self.get_action(state, explore=True)

                # 2. Execute action in environment
                next_state, reward, done, truncated, info = env.step(action)

                # 3. Update Q-value
                self.update(state, action, reward, next_state, done or truncated)

                total_reward += reward
                state = next_state
                steps += 1

                if done or truncated:
                    if done:
                        self.goals_reached += 1
                    break

            # 4. Decay exploration rate
            self.decay_epsilon()

            # 5. Record metrics
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
            self.epsilon_history.append(self.epsilon)

            if callback:
                callback(episode, total_reward, steps, self.epsilon)

        return self.get_training_metrics()

    def get_training_metrics(self):
        """Return a summary of training performance with research-backed metrics."""
        if not self.episode_rewards:
            return {'status': 'not_trained'}

        rewards = self.episode_rewards
        total_eps = len(rewards)

        # ── Policy Stability: consistency of action selection ──
        # Measures how often the agent picks the same action for identical states.
        # High stability = mature, reliable policy (research benchmark).
        policy_stability = self._compute_policy_stability()

        # ── Goal Achievement Rate: % of episodes reaching the goal ──
        goal_rate = round((self.goals_reached / total_eps * 100), 1) if total_eps > 0 else 0

        # ── State Coverage: % with sufficient visits (N≥10) ──
        # Research: a state needs N(s)≥50 visits for reliable Q-values.
        # We use N≥10 as a practical threshold for 3000-episode training.
        total_possible = 10 * 5 * 5 * 5 * 5 * 7  # 43,750 states
        well_covered = sum(1 for n in self.state_visit_counts.values() if n >= 10)
        coverage = round((well_covered / total_possible * 100), 2)

        return {
            'total_episodes': total_eps,
            'avg_reward': round(np.mean(rewards), 2),
            'avg_reward_last_100': round(np.mean(rewards[-100:]), 2),
            'best_reward': round(max(rewards), 2),
            'worst_reward': round(min(rewards), 2),
            'final_epsilon': round(self.epsilon, 4),
            'states_explored': len(self.states_visited),
            'q_table_size': len(self.q_table),
            'policy_stability': policy_stability,
            'goal_achievement_rate': goal_rate,
            'state_coverage': coverage,
            'convergence_delta': round(self.max_q_delta, 4),
            'episode_rewards': [round(r, 2) for r in rewards],
            'epsilon_history': [round(e, 4) for e in self.epsilon_history],
        }

    def _compute_policy_stability(self):
        """
        Compute policy stability: for each visited state, check if the
        best action is consistent (single dominant action).
        Returns percentage of stable states (0-100).
        """
        if not self.states_visited:
            return 0.0
        stable = 0
        sample_states = list(self.states_visited)[:500]  # Sample for performance
        for state in sample_states:
            q_vals = [self.q_table[(state, a)] for a in range(self.num_actions)]
            max_q = max(q_vals)
            # State is "stable" if only one action has the max Q-value
            num_best = sum(1 for q in q_vals if abs(q - max_q) < 0.01)
            if num_best == 1:
                stable += 1
        return round((stable / len(sample_states)) * 100, 1)

    def get_recommendation(self, state):
        """
        Get the agent's recommended action for the current state.
        Uses the learned Q-values (no exploration).

        Returns:
            dict: Action details and Q-value information
        """
        best_action = self._get_best_action(state)
        q_value = self.q_table[(state, best_action)]

        # Get all Q-values for this state
        all_q = [(a, self.q_table[(state, a)]) for a in range(self.num_actions)]
        all_q.sort(key=lambda x: x[1], reverse=True)

        return {
            'best_action': best_action,
            'q_value': round(q_value, 3),
            'top_5_actions': [
                {'action': a, 'q_value': round(q, 3)}
                for a, q in all_q[:5]
            ]
        }

    def reset_agent(self):
        """Reset the agent's learned knowledge."""
        self.q_table = defaultdict(lambda: self.optimistic_init)
        self.epsilon = EPSILON_START
        self.episode_rewards = []
        self.episode_steps = []
        self.epsilon_history = []
        self.states_visited = set()
        self.goals_reached = 0
        self.state_visit_counts = defaultdict(int)
        self.max_q_delta = 0.0
