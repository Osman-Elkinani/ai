"""
A* Search Agent — Informed Search for Optimal Workout Planning (Lecture 3).

A* Search finds the optimal path from the current fitness state to
the goal state by exploring the most promising workout sequences first.

Key concepts from Lecture 3:
  - f(n) = g(n) + h(n)
    - g(n): actual cost from start to node n (negative reward accumulated)
    - h(n): heuristic estimate of cost from n to goal (admissible)
  - A* is optimal when h(n) is admissible (never overestimates)
  - We use Graph Search to avoid revisiting states (Lecture 3)

Comparison with RL:
  - A* requires a model of the environment (deterministic transitions)
  - A* finds the optimal plan but cannot adapt to stochastic outcomes
  - RL learns from experience and handles stochasticity naturally
"""

import heapq
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MAX_SEARCH_DEPTH, SEARCH_BEAM_WIDTH, NUM_ACTIONS


class SearchNode:
    """
    Node in the A* search tree.
    
    Each node represents a state in the fitness journey, along with:
      - The path of actions taken to reach it
      - g(n): total reward accumulated (negated for min-heap)
      - h(n): heuristic estimate to goal
      - f(n) = g(n) + h(n): priority for expansion
    """

    def __init__(self, state, actions=None, g_cost=0.0, h_cost=0.0, parent=None):
        self.state = state
        self.actions = actions or []     # Sequence of action indices
        self.g_cost = g_cost             # Cost so far (negative total reward)
        self.h_cost = h_cost             # Heuristic estimate to goal
        self.f_cost = g_cost + h_cost    # f(n) = g(n) + h(n)
        self.parent = parent
        self.depth = len(self.actions)

    def __lt__(self, other):
        """For heap comparison — lower f_cost is better."""
        return self.f_cost < other.f_cost


class AStarSearchAgent:
    """
    A* Search agent that finds optimal multi-day workout plans.

    Uses the environment's deterministic transition model and
    heuristic function to search for the best sequence of
    workout + nutrition actions to reach the fitness goal.

    This implements Graph Search (Lecture 3) with a closed set
    to avoid revisiting states.
    """

    def __init__(self, env, max_depth=MAX_SEARCH_DEPTH,
                 beam_width=SEARCH_BEAM_WIDTH):
        """
        Initialize the A* search agent.

        Args:
            env: FitnessEnv instance (provides transition model & heuristic)
            max_depth: Maximum plan length (days to plan ahead)
            beam_width: Maximum nodes to expand (prevents explosion)
        """
        self.env = env
        self.max_depth = max_depth
        self.beam_width = beam_width

        # Search metrics
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.max_frontier_size = 0

    def search(self, start_state=None):
        """
        Run A* search from the current state to the goal.

        Algorithm (Lecture 3):
          1. Initialize frontier (priority queue) with start node
          2. Initialize closed set (visited states — Graph Search)
          3. While frontier is not empty:
             a. Pop node with lowest f(n) = g(n) + h(n)
             b. If goal reached → return solution
             c. If already visited → skip (Graph Search optimization)
             d. Add to closed set
             e. For each action:
                - Generate successor state
                - Compute g(n') and h(n')
                - Add to frontier if not visited

        Returns:
            dict: Search results including optimal plan and metrics
        """
        if start_state is None:
            start_state = self.env.state

        # ── Initialize search ──
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.max_frontier_size = 0

        h_start = self.env.state_distance_to_goal(start_state)
        start_node = SearchNode(state=start_state, g_cost=0, h_cost=h_start)

        # Priority queue (min-heap) — the FRONTIER
        frontier = [start_node]
        heapq.heapify(frontier)

        # Closed set — visited states (Graph Search, Lecture 3)
        closed = set()

        # Track the best solution found (in case we don't reach goal)
        best_node = start_node

        while frontier and self.nodes_expanded < self.beam_width:
            # ── Pop the most promising node ──
            current = heapq.heappop(frontier)

            # ── Goal test ──
            if self._is_goal(current.state):
                return self._build_result(current, 'goal_reached')

            # ── Graph Search: skip if already visited ──
            state_key = current.state
            if state_key in closed:
                continue
            closed.add(state_key)

            self.nodes_expanded += 1

            # ── Depth limit check ──
            if current.depth >= self.max_depth:
                if current.h_cost < best_node.h_cost:
                    best_node = current
                continue

            # ── Expand: generate successors for all actions ──
            for action in range(NUM_ACTIONS):
                # Use DETERMINISTIC transitions for search
                next_state = self.env.get_deterministic_next_state(
                    current.state, action
                )

                if next_state in closed:
                    continue

                # Calculate costs
                # g(n') = g(n) + step_cost
                # We use negative reward as cost (since A* minimizes)
                step_reward = self._estimate_reward(current.state, next_state, action)
                new_g = current.g_cost - step_reward  # Negate: more reward = less cost

                # h(n') = heuristic estimate to goal
                new_h = self.env.state_distance_to_goal(next_state)

                # Create successor node
                successor = SearchNode(
                    state=next_state,
                    actions=current.actions + [action],
                    g_cost=new_g,
                    h_cost=new_h,
                    parent=current,
                )

                heapq.heappush(frontier, successor)
                self.nodes_generated += 1

                # Track best node seen
                if new_h < best_node.h_cost:
                    best_node = successor

            self.max_frontier_size = max(self.max_frontier_size, len(frontier))

        # ── If goal not reached, return best partial plan ──
        return self._build_result(best_node, 'depth_limited')

    def _is_goal(self, state):
        """Check if a state satisfies the goal conditions."""
        fitness, energy, weight, muscle, fatigue, _ = state
        target = self.env.target_state
        goal_type = self.env.goal_type

        if goal_type == 'weight_loss':
            return (weight <= target.get('weight', 2) and
                    fitness >= target.get('fitness', 5))
        elif goal_type == 'muscle_gain':
            return (muscle >= target.get('muscle', 4) and
                    fitness >= target.get('fitness', 6))
        elif goal_type == 'general_fitness':
            return (fitness >= target.get('fitness', 7) and
                    muscle >= target.get('muscle', 3))
        elif goal_type == 'endurance':
            return fitness >= target.get('fitness', 8)
        return False

    def _estimate_reward(self, state, next_state, action):
        """
        Estimate the reward for a transition (used as step cost).
        Simplified version of the environment's reward function
        for deterministic planning.
        """
        old_f, old_e, old_w, old_m, old_fat, _ = state
        new_f, new_e, new_w, new_m, new_fat, _ = next_state

        reward = 0.0
        goal = self.env.goal_type

        if goal == 'weight_loss':
            reward += (old_w - new_w) * 3.0
            reward += (new_f - old_f) * 1.5
        elif goal == 'muscle_gain':
            reward += (new_m - old_m) * 3.0
            reward += (new_f - old_f) * 0.8
        elif goal == 'general_fitness':
            reward += (new_f - old_f) * 2.0
            reward += (new_m - old_m) * 1.5
        elif goal == 'endurance':
            reward += (new_f - old_f) * 3.5

        # Penalties
        if new_fat >= 4:
            reward -= 3.0
        if new_e <= 0:
            reward -= 2.0

        reward -= 0.1  # Time penalty

        return reward

    def _build_result(self, node, status):
        """Build the search result dictionary."""
        # Get action descriptions
        plan = []
        for i, action in enumerate(node.actions):
            desc = self.env.get_action_description(action)
            plan.append({
                'day': i + 1,
                'action': action,
                'workout': desc['workout'],
                'workout_emoji': desc['workout_emoji'],
                'nutrition': desc['nutrition'],
                'nutrition_emoji': desc['nutrition_emoji'],
                'description': desc['description'],
            })

        return {
            'status': status,
            'plan': plan,
            'plan_length': len(node.actions),
            'total_estimated_reward': round(-node.g_cost, 2),
            'final_heuristic': round(node.h_cost, 2),
            'nodes_expanded': self.nodes_expanded,
            'nodes_generated': self.nodes_generated,
            'max_frontier_size': self.max_frontier_size,
            'final_state': self.env._state_to_dict(node.state),
        }
