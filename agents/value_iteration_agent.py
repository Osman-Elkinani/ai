"""
Value Iteration Agent — Solving MDPs via Bellman Equations (Lecture 8).

Value Iteration computes the optimal value V*(s) for every state
by iteratively applying the Bellman Update:

    V*(s) = max_a [ R(s, a, s') + γ · V*(s') ]

Once V* converges, we extract the optimal policy:

    π*(s) = argmax_a [ R(s, a, s') + γ · V*(s') ]

Key concepts from Lecture 8:
  - Bellman Equation: recursive relationship between state values
  - Value Iteration: repeated application of Bellman updates
  - Policy Extraction: deriving the best action from V*
  - Convergence: values converge to unique optimal values

Difference from Q-Learning:
  - Value Iteration requires a MODEL (transition function T)
  - Value Iteration computes exact values (not approximate)
  - Q-Learning learns from experience without a model
"""

import itertools
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    FITNESS_LEVELS, ENERGY_LEVELS, WEIGHT_STATUSES,
    MUSCLE_LEVELS, FATIGUE_LEVELS, DISCOUNT_FACTOR,
    NUM_ACTIONS, WORKOUT_TYPES, NUTRITION_PLANS,
    NUM_NUTRITION
)


class ValueIterationAgent:
    """
    Value Iteration agent that solves the MDP exactly.

    Unlike Q-Learning (model-free), Value Iteration is MODEL-BASED:
    it needs the transition function T(s, a, s') to compute values.

    We simplify the state to 5 dimensions (drop day_of_week) to
    reduce from 43,750 to 6,250 states — making VI tractable.
    """

    def __init__(self, env, gamma=DISCOUNT_FACTOR,
                 num_iterations=50, threshold=0.01):
        """
        Args:
            env: FitnessEnv instance (provides T and R)
            gamma: Discount factor (γ = 0.95)
            num_iterations: Maximum Bellman update iterations
            threshold: Convergence threshold (stop when max change < θ)
        """
        self.env = env
        self.gamma = gamma
        self.num_iterations = num_iterations
        self.threshold = threshold

        # ── State Space (5D — day removed for efficiency) ──
        # 10 × 5 × 5 × 5 × 5 = 6,250 states
        self.states = list(itertools.product(
            range(FITNESS_LEVELS),   # fitness: 0-9
            range(ENERGY_LEVELS),    # energy: 0-4
            range(WEIGHT_STATUSES),  # weight: 0-4
            range(MUSCLE_LEVELS),    # muscle: 0-4
            range(FATIGUE_LEVELS),   # fatigue: 0-4
        ))

        # V*(s) — optimal value for each state
        self.V = {s: 0.0 for s in self.states}

        # π*(s) — optimal policy (best action for each state)
        self.policy = {}

        # Metrics
        self.convergence_history = []
        self.iterations_run = 0
        self.solve_time = 0

    def _to_6d(self, state_5d):
        """Add day=0 to convert 5D state to 6D for env methods."""
        return state_5d + (0,)

    def _to_5d(self, state_6d):
        """Remove day dimension from 6D state."""
        return state_6d[:5]

    def _compute_reward(self, old_6d, new_6d, action):
        """
        Compute reward for a state transition.
        Mirrors the environment's reward function.
        """
        old_f, old_e, old_w, old_m, old_fat, _ = old_6d
        new_f, new_e, new_w, new_m, new_fat, _ = new_6d

        workout_idx = action // NUM_NUTRITION
        workout_key = WORKOUT_TYPES[workout_idx]
        nutrition_key = NUTRITION_PLANS[action % NUM_NUTRITION]

        reward = 0.0
        goal = self.env.goal_type
        target = self.env.target_state

        if goal == 'weight_loss':
            reward += (old_w - new_w) * 3.0
            reward += (new_f - old_f) * 1.5
            if new_e >= 2:
                reward += 0.5
            if new_w <= target.get('weight', 2):
                reward += 2.0
        elif goal == 'muscle_gain':
            reward += (new_m - old_m) * 3.0
            if workout_key == 'strength_training' and nutrition_key == 'high_protein':
                reward += 1.5
            reward += (new_f - old_f) * 0.8
            if new_m >= target.get('muscle', 4):
                reward += 2.0
        elif goal == 'general_fitness':
            reward += (new_f - old_f) * 2.0
            reward += (new_m - old_m) * 1.5
            if old_w > 2:
                reward += (old_w - new_w) * 1.0
            reward += (new_e - old_e) * 0.5
            if new_f >= target.get('fitness', 7):
                reward += 1.5
        elif goal == 'endurance':
            reward += (new_f - old_f) * 3.5
            if workout_key in ['jogging', 'swimming', 'cycling']:
                reward += 1.0
            reward += (new_e - old_e) * 0.5

        # Universal penalties
        if new_fat >= 4:
            reward -= 3.0
        elif new_fat >= 3:
            reward -= 1.5
        if new_e <= 0:
            reward -= 2.0
        elif new_e <= 1:
            reward -= 0.5
        reward -= 0.1

        return reward

    def solve(self, callback=None):
        """
        Run Value Iteration (Lecture 8).

        Algorithm:
          1. Initialize V(s) = 0 for all states
          2. Repeat until convergence:
             For each state s:
               V(s) ← max_a [ R(s, a, s') + γ · V(s') ]
          3. Extract optimal policy:
             π(s) = argmax_a [ R(s, a, s') + γ · V(s') ]

        The key insight (Bellman Equation):
          The value of a state equals the best immediate reward
          plus the discounted value of the best next state.

        Returns:
            dict: Solution metrics
        """
        start_time = time.time()

        # ── Step 1: Initialize V(s) = 0 ──
        V = {s: 0.0 for s in self.states}
        self.convergence_history = []

        num_states = len(self.states)

        # ── Step 2: Iterate Bellman Updates ──
        for iteration in range(self.num_iterations):
            V_new = {}
            for s in self.states:
                full_s = self._to_6d(s)
                best_value = float('-inf')

                # Try all 28 actions, pick the best
                for a in range(NUM_ACTIONS):
                    # T(s, a) → s' (deterministic transition)
                    next_full = self.env.get_deterministic_next_state(full_s, a)
                    next_5d = self._to_5d(next_full)

                    # R(s, a, s')
                    r = self._compute_reward(full_s, next_full, a)

                    # Bellman: R + γ · V(s')
                    value = r + self.gamma * V[next_5d]
                    best_value = max(best_value, value)

                V_new[s] = best_value

            # ── Check Convergence ──
            delta = max(abs(V_new[s] - V[s]) for s in self.states)
            V = V_new
            self.convergence_history.append(round(delta, 4))

            if callback:
                callback(iteration, delta)

            # Stop if values converged (change < threshold)
            if delta < self.threshold:
                break

        self.V = V
        self.iterations_run = len(self.convergence_history)

        # ── Step 3: Policy Extraction (Lecture 8) ──
        # π*(s) = argmax_a [ R(s,a,s') + γ · V*(s') ]
        self._extract_policy()

        self.solve_time = round(time.time() - start_time, 2)
        return self.get_metrics()

    def _extract_policy(self):
        """
        Policy Extraction (Lecture 8).

        Given optimal values V*(s), compute the best action:
          π*(s) = argmax_a [ R(s, a, s') + γ · V*(s') ]
        """
        self.policy = {}
        for s in self.states:
            full_s = self._to_6d(s)
            best_action = 0
            best_value = float('-inf')

            for a in range(NUM_ACTIONS):
                next_full = self.env.get_deterministic_next_state(full_s, a)
                next_5d = self._to_5d(next_full)
                r = self._compute_reward(full_s, next_full, a)
                value = r + self.gamma * self.V[next_5d]

                if value > best_value:
                    best_value = value
                    best_action = a

            self.policy[s] = best_action

    def get_action(self, state):
        """Get the optimal action for a 6D state."""
        state_5d = self._to_5d(state)
        return self.policy.get(state_5d, 0)

    def get_recommendation(self, state):
        """Get recommendation with value information."""
        state_5d = self._to_5d(state)
        best_action = self.policy.get(state_5d, 0)
        v_value = self.V.get(state_5d, 0)

        # Get top 5 actions by value
        full_s = self._to_6d(state_5d)
        action_values = []
        for a in range(NUM_ACTIONS):
            next_full = self.env.get_deterministic_next_state(full_s, a)
            next_5d = self._to_5d(next_full)
            r = self._compute_reward(full_s, next_full, a)
            val = r + self.gamma * self.V.get(next_5d, 0)
            action_values.append((a, round(val, 3)))

        action_values.sort(key=lambda x: x[1], reverse=True)

        return {
            'best_action': best_action,
            'v_value': round(v_value, 3),
            'top_5_actions': [
                {'action': a, 'q_value': v}
                for a, v in action_values[:5]
            ]
        }

    def get_metrics(self):
        """Return solution metrics."""
        # Compute value statistics
        values = list(self.V.values())
        return {
            'iterations': self.iterations_run,
            'converged': self.convergence_history[-1] < self.threshold if self.convergence_history else False,
            'final_delta': self.convergence_history[-1] if self.convergence_history else 0,
            'solve_time': self.solve_time,
            'num_states': len(self.states),
            'avg_value': round(sum(values) / len(values), 2),
            'max_value': round(max(values), 2),
            'min_value': round(min(values), 2),
            'convergence_history': self.convergence_history,
        }
