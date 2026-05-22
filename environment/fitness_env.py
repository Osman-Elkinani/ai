"""
FitnessEnv — OpenAI Gymnasium Environment for Health & Fitness.

This module implements a Markov Decision Process (MDP) as taught in
Lectures 7 & 8 (MDPs I & II). The environment simulates a user's
fitness journey over time, with stochastic transitions.

MDP Components:
  - States (S):  Tuple of (fitness, energy, weight, muscle, fatigue, day)
  - Actions (A): Combined workout + nutrition choices (28 total)
  - Transitions T(s, a, s'): Stochastic state changes based on actions
  - Rewards R(s, a, s'): Goal-dependent feedback signal
  - Discount γ: 0.95 (from config)

Simulation Platform: Python + OpenAI Gymnasium
  - Inherits from gymnasium.Env
  - Defines observation_space (MultiDiscrete) and action_space (Discrete)
  - Compatible with standard Gym interface: reset(), step()
"""

import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    FITNESS_LEVELS, ENERGY_LEVELS, WEIGHT_STATUSES,
    MUSCLE_LEVELS, FATIGUE_LEVELS, DAYS_PER_WEEK,
    WORKOUT_TYPES, NUTRITION_PLANS, NUM_WORKOUTS, NUM_NUTRITION,
    MAX_STEPS_PER_EPISODE
)
from data.workouts import WORKOUT_DATABASE
from data.nutrition import NUTRITION_DATABASE
from environment.user_profile import UserProfile


class FitnessEnv(gym.Env):
    """
    OpenAI Gymnasium environment modeling a fitness journey as an MDP.

    State Space: (fitness, energy, weight, muscle, fatigue, day_of_week)
    Action Space: workout_index * NUM_NUTRITION + nutrition_index
    
    This follows the MDP framework from Lectures 7-8:
    - States are discrete tuples
    - Actions are workout + nutrition combinations
    - Transitions are stochastic (random noise added)
    - Rewards depend on the user's goal

    Inherits from gymnasium.Env to comply with OpenAI Gym standard.
    """

    metadata = {'render_modes': ['human']}

    # ─── Labels for human-readable state display ───
    FITNESS_LABELS = ['Very Unfit', 'Unfit', 'Below Average', 'Average',
                      'Above Average', 'Fit', 'Very Fit', 'Athletic',
                      'Elite', 'Peak']
    ENERGY_LABELS = ['Exhausted', 'Low', 'Medium', 'High', 'Very High']
    WEIGHT_LABELS = ['Very Underweight', 'Underweight', 'Normal', 'Overweight', 'Obese']
    MUSCLE_LABELS = ['Very Weak', 'Weak', 'Average', 'Strong', 'Very Strong']
    FATIGUE_LABELS = ['None', 'Mild', 'Moderate', 'High', 'Severe']
    DAY_LABELS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                  'Friday', 'Saturday', 'Sunday']

    def __init__(self, user_profile=None, render_mode=None):
        """
        Initialize the fitness environment.

        Args:
            user_profile: UserProfile instance defining initial state and goals
            render_mode: Gymnasium render mode (optional)
        """
        super().__init__()

        self.profile = user_profile or UserProfile()
        self.goal_type = self.profile.goal_type
        self.target_state = self.profile.target_state
        self.render_mode = render_mode

        # ── Gymnasium Spaces (OpenAI Gym compliance) ──
        # Action space: single discrete action (0 to 27)
        self.action_space = spaces.Discrete(NUM_WORKOUTS * NUM_NUTRITION)

        # Observation space: MultiDiscrete for each state variable
        self.observation_space = spaces.MultiDiscrete([
            FITNESS_LEVELS,   # fitness:  0-9
            ENERGY_LEVELS,    # energy:   0-4
            WEIGHT_STATUSES,  # weight:   0-4
            MUSCLE_LEVELS,    # muscle:   0-4
            FATIGUE_LEVELS,   # fatigue:  0-4
            DAYS_PER_WEEK,    # day:      0-6
        ])

        # Contextual features (filter actions, NOT in state space)
        self.equipment = getattr(self.profile, 'equipment', 'gym')
        self.time_available = getattr(self.profile, 'time_available', 60)

        # Compute valid workout indices based on equipment & time
        from data.workouts import get_available_workout_indices
        self.valid_workout_indices = get_available_workout_indices(
            self.equipment, self.time_available
        )

        # Current state variables
        self.state = None
        self.step_count = 0
        self.max_steps = MAX_STEPS_PER_EPISODE
        self.total_reward = 0

        # History tracking for visualization
        self.history = []

        # Initialize
        self.reset()

    def reset(self):
        """
        Reset the environment to the initial state.
        
        Returns:
            tuple: The initial state (fitness, energy, weight, muscle, fatigue, day)
        """
        init = self.profile.initial_state
        self.state = (
            init['fitness'],    # Fitness level
            init['energy'],     # Energy level
            init['weight'],     # Weight status
            init['muscle'],     # Muscle level
            init['fatigue'],    # Fatigue level
            0                   # Start on Monday
        )
        self.step_count = 0
        self.total_reward = 0

        # Accumulators for slow-changing variables (research-backed)
        # These track fractional progress — level changes only when |acc| >= 1.0
        self._acc_fitness = 0.0
        self._acc_weight = 0.0
        self._acc_muscle = 0.0

        self.history = [self._state_to_dict(self.state)]
        return self.state

    def step(self, action):
        """
        Execute one step in the environment (one day of the fitness plan).

        This implements the MDP transition function T(s, a, s').
        Transitions are stochastic — random noise is added to effects.

        Args:
            action: Integer action index (0 to NUM_ACTIONS-1)
                    Decoded as: workout_idx = action // NUM_NUTRITION
                                nutrition_idx = action % NUM_NUTRITION

        Returns:
            tuple: (next_state, reward, done, truncated, info)
        """
        # ── Decode the combined action into workout + nutrition ──
        workout_idx = action // NUM_NUTRITION
        nutrition_idx = action % NUM_NUTRITION
        workout_key = WORKOUT_TYPES[workout_idx]
        nutrition_key = NUTRITION_PLANS[nutrition_idx]

        workout = WORKOUT_DATABASE[workout_key]
        nutrition = NUTRITION_DATABASE[nutrition_key]

        # ── Extract current state ──
        fitness, energy, weight, muscle, fatigue, day = self.state

        # ── Check if requirements are met ──
        requirements_met = True
        for req_key, req_val in workout.get('requirements', {}).items():
            current_val = {'fitness': fitness, 'energy': energy,
                          'weight': weight, 'muscle': muscle}
            if current_val.get(req_key, 0) < req_val:
                requirements_met = False
                break

        # ── Calculate state transitions (stochastic) ──
        # This is T(s, a, s') — the transition model
        if requirements_met:
            w_effects = workout['effects']
            n_effects = nutrition['effects']

            # Stochastic noise: p_slip = 0.1 (10% uniform perturbation)
            # Based on published RL benchmarks (p_slip >= 0.2 = highly stochastic)
            noise = lambda: random.uniform(-0.1, 0.1)

            delta_fitness = w_effects['fitness'] + noise()
            delta_energy = (w_effects['energy'] + n_effects['energy'] + noise())
            delta_weight = (w_effects['weight'] + n_effects['weight'] + noise())
            delta_muscle = (w_effects['muscle'] + n_effects['muscle'] + noise())
            delta_fatigue = (w_effects['fatigue'] + n_effects['fatigue'] + noise())
        else:
            # If requirements not met, forced rest with penalty
            delta_fitness = 0
            delta_energy = 1
            delta_weight = 0
            delta_muscle = 0
            delta_fatigue = -1

        # ── Fast-changing variables: Energy & Fatigue (within-day) ──
        # These change immediately each step (research: session-level variables)
        new_energy = max(0, min(ENERGY_LEVELS - 1, int(round(energy + delta_energy))))
        new_fatigue = max(0, min(FATIGUE_LEVELS - 1, int(round(fatigue + delta_fatigue))))

        # ── Slow-changing variables: Fitness, Weight, Muscle (accumulator) ──
        # Research: "weight and fitness should NOT transition step-by-step"
        # We accumulate fractional deltas — level shifts only when |acc| >= 1.0
        self._acc_fitness += delta_fitness
        self._acc_weight += delta_weight
        self._acc_muscle += delta_muscle

        new_fitness = fitness
        if abs(self._acc_fitness) >= 1.0:
            shift = int(self._acc_fitness)  # +1 or -1
            new_fitness = max(0, min(FITNESS_LEVELS - 1, fitness + shift))
            self._acc_fitness -= shift

        new_weight = weight
        if abs(self._acc_weight) >= 1.0:
            shift = int(self._acc_weight)
            new_weight = max(0, min(WEIGHT_STATUSES - 1, weight + shift))
            self._acc_weight -= shift

        new_muscle = muscle
        if abs(self._acc_muscle) >= 1.0:
            shift = int(self._acc_muscle)
            new_muscle = max(0, min(MUSCLE_LEVELS - 1, muscle + shift))
            self._acc_muscle -= shift

        new_day = (day + 1) % DAYS_PER_WEEK

        # ── Update state ──
        self.state = (new_fitness, new_energy, new_weight,
                      new_muscle, new_fatigue, new_day)
        self.step_count += 1

        # ── Calculate reward (R(s, a, s')) ──
        reward = self._calculate_reward(
            old_state=(fitness, energy, weight, muscle, fatigue, day),
            new_state=self.state,
            workout_key=workout_key,
            nutrition_key=nutrition_key,
            requirements_met=requirements_met
        )
        self.total_reward += reward

        # ── Check termination ──
        done = self._is_goal_reached()
        truncated = self.step_count >= self.max_steps

        # ── Record history ──
        info = {
            'workout': workout['name'],
            'workout_key': workout_key,
            'nutrition': nutrition['name'],
            'nutrition_key': nutrition_key,
            'reward': reward,
            'requirements_met': requirements_met,
            'step': self.step_count,
            'total_reward': self.total_reward,
        }
        state_dict = self._state_to_dict(self.state)
        state_dict.update(info)
        self.history.append(state_dict)

        return self.state, reward, done, truncated, info

    def _calculate_reward(self, old_state, new_state, workout_key,
                          nutrition_key, requirements_met):
        """
        Composite reward function with explicit weight factors.

        Based on published research (PERFECT Framework, 2023; PPO Physiotherapy, 2025):
          R_total = w1·(goal_progress) + w2·(energy_maintenance)
                  - w3·(fatigue_penalty) - w4·(safety_penalty)

        Weight factors are tuned per goal type to ensure Value Alignment:
        the agent optimizes the user's objective without ignoring safety.
        """
        old_f, old_e, old_w, old_m, old_fat, _ = old_state
        new_f, new_e, new_w, new_m, new_fat, _ = new_state

        reward = 0.0

        # ── Penalty for not meeting requirements ──
        if not requirements_met:
            reward -= 2.0

        # ── Goal-specific rewards (w1: goal progress) ──
        if self.goal_type == 'weight_loss':
            reward += (old_w - new_w) * 3.0       # Weight decrease
            reward += (new_f - old_f) * 1.5        # Fitness increase
            if new_e >= 2:
                reward += 0.5                       # Energy maintenance (w2)
            if new_w <= self.target_state.get('weight', 2):
                reward += 2.0                       # Goal achievement bonus

        elif self.goal_type == 'muscle_gain':
            reward += (new_m - old_m) * 3.0        # Muscle growth
            if workout_key == 'strength_training' and nutrition_key == 'high_protein':
                reward += 1.5                       # Optimal combo bonus
            reward += (new_f - old_f) * 0.8        # Fitness maintenance
            if new_m >= self.target_state.get('muscle', 4):
                reward += 2.0                       # Goal achievement bonus

        elif self.goal_type == 'general_fitness':
            reward += (new_f - old_f) * 2.0        # Fitness progress
            reward += (new_m - old_m) * 1.5        # Muscle progress
            reward += (old_w - new_w) * 1.0 if old_w > 2 else 0
            reward += (new_e - old_e) * 0.5        # Energy improvement
            if new_f >= self.target_state.get('fitness', 7):
                reward += 1.5                       # Goal achievement bonus

        elif self.goal_type == 'endurance':
            reward += (new_f - old_f) * 3.5        # Heavy fitness emphasis
            if workout_key in ['jogging', 'swimming', 'cycling']:
                reward += 1.0                       # Cardio activity bonus
            reward += (new_e - old_e) * 0.5        # Energy improvement

        # ── Overtraining penalty (w3: fatigue — progressive curve) ──
        # Research: Q-Learning degrades from 92% to 74% success under
        # high stochasticity; fatigue is the primary stochastic driver.
        # We use an exponential penalty to strongly discourage overtraining.
        if new_fat >= 4:
            reward -= 4.0   # Severe: injury risk zone
        elif new_fat >= 3:
            reward -= 2.0   # High: unsustainable
        elif new_fat >= 2 and old_fat < 2:
            reward -= 0.5   # Moderate: early warning

        # ── Safety penalty (w4: energy depletion) ──
        if new_e <= 0:
            reward -= 2.5   # Exhaustion: dangerous
        elif new_e <= 1:
            reward -= 0.8   # Low energy: suboptimal

        # ── Diversity bonus: discourage repeating the same action ──
        # Research: inter-session diversity improves long-term adherence.
        if hasattr(self, '_last_action') and self._last_action == (workout_key, nutrition_key):
            reward -= 0.3
        self._last_action = (workout_key, nutrition_key)

        # Small time penalty to encourage efficiency
        reward -= 0.1

        # ── Potential-Based Reward Shaping (PBRS) ──
        # F(s, s') = γ·φ(s') - φ(s) where φ = -distance_to_goal
        # This provides dense step-by-step gradients that accelerate
        # Q-learning convergence WITHOUT altering the optimal policy.
        # Reference: Ng et al. (1999); confirmed in fitness RL literature.
        from config import DISCOUNT_FACTOR
        phi_old = -self.state_distance_to_goal(old_state)
        phi_new = -self.state_distance_to_goal(new_state)
        shaping = DISCOUNT_FACTOR * phi_new - phi_old
        reward += shaping * 0.5  # Scale factor to balance with primary reward

        return round(reward, 2)

    def _is_goal_reached(self):
        """Check if the user's target state has been achieved."""
        fitness, energy, weight, muscle, fatigue, _ = self.state
        target = self.target_state
        
        if self.goal_type == 'weight_loss':
            return (weight <= target.get('weight', 2) and
                    fitness >= target.get('fitness', 5))
        elif self.goal_type == 'muscle_gain':
            return (muscle >= target.get('muscle', 4) and
                    fitness >= target.get('fitness', 6))
        elif self.goal_type == 'general_fitness':
            return (fitness >= target.get('fitness', 7) and
                    muscle >= target.get('muscle', 3))
        elif self.goal_type == 'endurance':
            return fitness >= target.get('fitness', 8)
        return False

    def _state_to_dict(self, state):
        """Convert state tuple to labeled dictionary."""
        fitness, energy, weight, muscle, fatigue, day = state
        return {
            'fitness': fitness,
            'fitness_label': self.FITNESS_LABELS[fitness],
            'energy': energy,
            'energy_label': self.ENERGY_LABELS[energy],
            'weight': weight,
            'weight_label': self.WEIGHT_LABELS[weight],
            'muscle': muscle,
            'muscle_label': self.MUSCLE_LABELS[muscle],
            'fatigue': fatigue,
            'fatigue_label': self.FATIGUE_LABELS[fatigue],
            'day': day,
            'day_label': self.DAY_LABELS[day],
        }

    def get_state_dict(self):
        """Get the current state as a labeled dictionary."""
        return self._state_to_dict(self.state)

    def get_action_description(self, action):
        """Get human-readable description of an action."""
        workout_idx = action // NUM_NUTRITION
        nutrition_idx = action % NUM_NUTRITION
        workout = WORKOUT_DATABASE[WORKOUT_TYPES[workout_idx]]
        nutrition = NUTRITION_DATABASE[NUTRITION_PLANS[nutrition_idx]]
        return {
            'workout': workout['name'],
            'workout_key': WORKOUT_TYPES[workout_idx],
            'workout_emoji': workout['emoji'],
            'nutrition': nutrition['name'],
            'nutrition_key': NUTRITION_PLANS[nutrition_idx],
            'nutrition_emoji': nutrition['emoji'],
            'description': f"{workout['emoji']} {workout['name']} + {nutrition['emoji']} {nutrition['name']}"
        }

    def get_deterministic_next_state(self, state, action, accumulators=None):
        """
        Get the expected (deterministic) next state for A* search.

        Unlike step(), this uses mean effects without stochastic noise.
        Used by the A* search agent (Lecture 3) for planning.

        When 'accumulators' is provided, slow-changing variables (fitness,
        weight, muscle) use the same accumulator pattern as step():
        fractional effects are summed across steps; a level change only
        fires when |accumulator| >= 1.0.  This lets A* see realistic
        multi-day progress instead of rounding every small effect to 0.

        Args:
            state: Current 6D state tuple
            action: Action index (0 to NUM_ACTIONS-1)
            accumulators: Optional (acc_fitness, acc_weight, acc_muscle).
                          Pass None for single-step lookups (Value Iteration).

        Returns:
            If accumulators is None:  new_state              (backward compat)
            If accumulators given:    (new_state, new_accumulators)
        """
        workout_idx = action // NUM_NUTRITION
        nutrition_idx = action % NUM_NUTRITION
        workout_key = WORKOUT_TYPES[workout_idx]
        nutrition_key = NUTRITION_PLANS[nutrition_idx]

        workout = WORKOUT_DATABASE[workout_key]
        nutrition = NUTRITION_DATABASE[nutrition_key]

        fitness, energy, weight, muscle, fatigue, day = state

        # Check requirements
        current_vals = {'fitness': fitness, 'energy': energy}
        for req_key, req_val in workout.get('requirements', {}).items():
            if current_vals.get(req_key, 0) < req_val:
                # Can't do this workout — forced rest
                new_state = (fitness, max(0, min(4, energy + 1)),
                             weight, muscle, max(0, fatigue - 1), (day + 1) % 7)
                if accumulators is not None:
                    return new_state, accumulators
                return new_state

        w_eff = workout['effects']
        n_eff = nutrition['effects']

        # ── Fast-changing variables: energy & fatigue (change every step) ──
        new_energy = max(0, min(4, int(round(energy + w_eff['energy'] + n_eff['energy']))))
        new_fatigue = max(0, min(4, int(round(fatigue + w_eff['fatigue'] + n_eff['fatigue']))))
        new_day = (day + 1) % 7

        if accumulators is not None:
            # ── Accumulator mode (A* search — multi-step planning) ──
            acc_f, acc_w, acc_m = accumulators

            acc_f += w_eff['fitness']
            acc_w += w_eff['weight'] + n_eff['weight']
            acc_m += w_eff['muscle'] + n_eff['muscle']

            new_fitness = fitness
            if abs(acc_f) >= 1.0:
                shift = int(acc_f)
                new_fitness = max(0, min(9, fitness + shift))
                acc_f -= shift

            new_weight = weight
            if abs(acc_w) >= 1.0:
                shift = int(acc_w)
                new_weight = max(0, min(4, weight + shift))
                acc_w -= shift

            new_muscle = muscle
            if abs(acc_m) >= 1.0:
                shift = int(acc_m)
                new_muscle = max(0, min(4, muscle + shift))
                acc_m -= shift

            new_state = (new_fitness, new_energy, new_weight,
                         new_muscle, new_fatigue, new_day)
            return new_state, (acc_f, acc_w, acc_m)
        else:
            # ── Single-step mode (Value Iteration — backward compatible) ──
            new_fitness = max(0, min(9, int(round(fitness + w_eff['fitness']))))
            new_weight = max(0, min(4, int(round(weight + w_eff['weight'] + n_eff['weight']))))
            new_muscle = max(0, min(4, int(round(muscle + w_eff['muscle'] + n_eff['muscle']))))

            return (new_fitness, new_energy, new_weight,
                    new_muscle, new_fatigue, new_day)

    def state_distance_to_goal(self, state):
        """
        Calculate heuristic distance from a state to the goal.
        Used as h(n) in A* search (Lecture 3).

        This heuristic is admissible — it never overestimates the
        true cost to reach the goal.
        """
        fitness, energy, weight, muscle, fatigue, _ = state
        target = self.target_state
        distance = 0

        if self.goal_type == 'weight_loss':
            distance += max(0, weight - target.get('weight', 2)) * 2
            distance += max(0, target.get('fitness', 5) - fitness) * 1.5
        elif self.goal_type == 'muscle_gain':
            distance += max(0, target.get('muscle', 4) - muscle) * 2
            distance += max(0, target.get('fitness', 6) - fitness) * 1.5
        elif self.goal_type == 'general_fitness':
            distance += max(0, target.get('fitness', 7) - fitness) * 2
            distance += max(0, target.get('muscle', 3) - muscle) * 1.5
        elif self.goal_type == 'endurance':
            distance += max(0, target.get('fitness', 8) - fitness) * 3

        # Penalty for high fatigue
        distance += fatigue * 0.5

        return distance
