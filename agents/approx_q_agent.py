"""
Approximate Q-Learning Agent — Feature-Based RL (Lecture 10: RL II).

When the state space is too large for a full Q-table, we use
Approximate Q-Learning with feature-based state representations.

Key concept (Lecture 10):
  Q(s, a) ≈ w₁·f₁(s, a) + w₂·f₂(s, a) + ... + wₙ·fₙ(s, a)

Instead of storing Q-values for every (s, a) pair, we learn a
weight vector w that, combined with feature functions f(s, a),
approximates the Q-value.

Update rule:
  difference = [r + γ · max_a' Q(s', a')] - Q(s, a)
  wᵢ ← wᵢ + α · difference · fᵢ(s, a)

This allows generalization across similar states — the agent can
make good decisions even for states it has never visited.
"""

import random
import numpy as np
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    APPROX_LEARNING_RATE, DISCOUNT_FACTOR, EPSILON_START,
    EPSILON_END, EPSILON_DECAY, NUM_EPISODES,
    MAX_STEPS_PER_EPISODE, NUM_ACTIONS, NUM_WORKOUTS,
    NUM_NUTRITION, FITNESS_LEVELS, ENERGY_LEVELS,
    WEIGHT_STATUSES, MUSCLE_LEVELS, FATIGUE_LEVELS
)


class ApproximateQLearningAgent:
    """
    Feature-based Approximate Q-Learning agent.

    Instead of maintaining a massive Q-table, this agent learns
    weights for feature functions that generalize across states.
    This is especially important when the state space is large
    (our environment has ~43,750 states × 28 actions = ~1.2M entries).
    """

    def __init__(self, num_actions=NUM_ACTIONS, learning_rate=APPROX_LEARNING_RATE,
                 discount_factor=DISCOUNT_FACTOR, epsilon_start=EPSILON_START,
                 epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY,
                 goal_type='weight_loss', target_state=None):
        """
        Initialize the Approximate Q-Learning agent.

        Args:
            num_actions: Number of possible actions
            learning_rate: α for weight updates (smaller than tabular)
            discount_factor: γ for future reward discounting
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate per episode
            goal_type: User's fitness goal
            target_state: Target state dict
        """
        self.num_actions = num_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.goal_type = goal_type
        self.target_state = target_state or {'fitness': 7, 'weight': 2, 'muscle': 3}

        # Weight vector — one weight per feature
        # Initialized to small random values
        self.num_features = 12
        self.weights = np.random.uniform(-0.01, 0.01, self.num_features)

        # ── Training Metrics ──
        self.episode_rewards = []
        self.episode_steps = []
        self.epsilon_history = []

    def _extract_features(self, state, action):
        """
        Extract feature vector f(s, a) from a state-action pair.

        Feature engineering is crucial for Approximate Q-Learning.
        We design features that capture important aspects of the
        state and how the action relates to the goal.

        Features (Lecture 10):
          f1: Normalized fitness level
          f2: Normalized energy level
          f3: Fitness gap to goal (how far from target)
          f4: Muscle gap to goal
          f5: Weight gap to goal
          f6: Fatigue level (penalty feature)
          f7: Is this a cardio workout? (1/0)
          f8: Is this strength training? (1/0)
          f9: Is this a rest day? (1/0)
          f10: Is nutrition high protein? (1/0)
          f11: Is nutrition caloric deficit? (1/0)
          f12: Bias term (always 1)
        """
        fitness, energy, weight, muscle, fatigue, day = state
        workout_idx = action // NUM_NUTRITION
        nutrition_idx = action % NUM_NUTRITION

        features = np.zeros(self.num_features)

        # ── State features (normalized to [0, 1]) ──
        features[0] = fitness / (FITNESS_LEVELS - 1)     # Fitness level
        features[1] = energy / (ENERGY_LEVELS - 1)       # Energy level

        # ── Goal proximity features ──
        target_fitness = self.target_state.get('fitness', 7)
        target_muscle = self.target_state.get('muscle', 3)
        target_weight = self.target_state.get('weight', 2)

        features[2] = max(0, target_fitness - fitness) / FITNESS_LEVELS  # Fitness gap
        features[3] = max(0, target_muscle - muscle) / MUSCLE_LEVELS     # Muscle gap
        features[4] = abs(weight - target_weight) / WEIGHT_STATUSES      # Weight gap
        features[5] = fatigue / (FATIGUE_LEVELS - 1)                     # Fatigue

        # ── Action features (one-hot style) ──
        features[6] = 1.0 if workout_idx in [2, 5, 6] else 0.0  # Cardio (jogging, swimming, cycling)
        features[7] = 1.0 if workout_idx == 4 else 0.0          # Strength
        features[8] = 1.0 if workout_idx == 0 else 0.0          # Rest
        features[9] = 1.0 if nutrition_idx == 3 else 0.0        # High protein
        features[10] = 1.0 if nutrition_idx == 0 else 0.0       # Deficit

        # ── Bias ──
        features[11] = 1.0

        return features

    def get_q_value(self, state, action):
        """
        Compute approximate Q-value: Q(s, a) ≈ w · f(s, a).

        This is a dot product between the weight vector and
        the feature vector extracted from the state-action pair.
        """
        features = self._extract_features(state, action)
        return np.dot(self.weights, features)

    def get_action(self, state, explore=True):
        """
        Select action using ε-greedy policy with approximate Q-values.
        """
        if explore and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return self._get_best_action(state)

    def _get_best_action(self, state):
        """Return action with highest approximate Q-value."""
        q_values = [self.get_q_value(state, a) for a in range(self.num_actions)]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if abs(q - max_q) < 1e-10]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        """
        Update weights using the Approximate Q-Learning rule (Lecture 10).

        difference = [r + γ · max_a' Q(s', a')] - Q(s, a)
        wᵢ ← wᵢ + α · difference · fᵢ(s, a)

        This updates ALL weights simultaneously, scaled by
        each feature's contribution.
        """
        # Current Q-value
        current_q = self.get_q_value(state, action)

        # TD Target
        if done:
            td_target = reward
        else:
            max_next_q = max(
                self.get_q_value(next_state, a)
                for a in range(self.num_actions)
            )
            td_target = reward + self.gamma * max_next_q

        # TD Error (difference)
        difference = td_target - current_q

        # Feature vector for current (s, a)
        features = self._extract_features(state, action)

        # ── Weight Update ──
        # wᵢ ← wᵢ + α · difference · fᵢ(s, a)
        self.weights += self.alpha * difference * features

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train(self, env, num_episodes=NUM_EPISODES, callback=None):
        """
        Train the approximate Q-learning agent.

        Same training loop as tabular Q-learning, but Q-values are
        computed from features instead of looked up in a table.
        """
        self.episode_rewards = []
        self.episode_steps = []
        self.epsilon_history = []

        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0

            for step in range(MAX_STEPS_PER_EPISODE):
                action = self.get_action(state, explore=True)
                next_state, reward, done, truncated, info = env.step(action)
                self.update(state, action, reward, next_state, done or truncated)

                total_reward += reward
                state = next_state
                steps += 1

                if done or truncated:
                    break

            self.decay_epsilon()
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
            self.epsilon_history.append(self.epsilon)

            if callback:
                callback(episode, total_reward, steps, self.epsilon)

        return self.get_training_metrics()

    def get_training_metrics(self):
        """Return training performance summary."""
        if not self.episode_rewards:
            return {'status': 'not_trained'}

        rewards = self.episode_rewards
        return {
            'total_episodes': len(rewards),
            'avg_reward': round(np.mean(rewards), 2),
            'avg_reward_last_100': round(np.mean(rewards[-100:]), 2),
            'best_reward': round(max(rewards), 2),
            'worst_reward': round(min(rewards), 2),
            'final_epsilon': round(self.epsilon, 4),
            'weights': [round(w, 4) for w in self.weights],
            'episode_rewards': [round(r, 2) for r in rewards],
            'epsilon_history': [round(e, 4) for e in self.epsilon_history],
        }

    def get_recommendation(self, state):
        """Get the agent's best action recommendation."""
        best_action = self._get_best_action(state)
        q_value = self.get_q_value(state, best_action)

        all_q = [(a, self.get_q_value(state, a)) for a in range(self.num_actions)]
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
        """Reset the agent."""
        self.weights = np.random.uniform(-0.01, 0.01, self.num_features)
        self.epsilon = EPSILON_START
        self.episode_rewards = []
        self.episode_steps = []
        self.epsilon_history = []
