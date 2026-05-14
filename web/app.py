"""
Flask Web Application — Dashboard API Server.

Provides REST API endpoints for the web dashboard to:
  - Configure user profiles and goals
  - Train Q-Learning and Approximate Q-Learning agents
  - Run A* Search for optimal plans
  - Simulate steps in the environment
  - Get recommendations and metrics
"""

from flask import Flask, render_template, jsonify, request
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    WEB_HOST, WEB_PORT, DEBUG, NUM_EPISODES,
    WORKOUT_TYPES, NUTRITION_PLANS
)
from environment.fitness_env import FitnessEnv
from environment.user_profile import UserProfile
from agents.q_learning_agent import QLearningAgent
from agents.approx_q_agent import ApproximateQLearningAgent
from agents.search_agent import AStarSearchAgent
from agents.value_iteration_agent import ValueIterationAgent

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════════
# Global State (for simplicity in this educational project)
# ═══════════════════════════════════════════════════════════════
env = None                  # FitnessEnv instance
q_agent = None              # Tabular Q-Learning agent
approx_agent = None         # Approximate Q-Learning agent
search_agent = None         # A* Search agent
vi_agent = None             # Value Iteration agent
current_profile = None      # Active user profile
simulation_log = []         # Log of simulation steps


def _initialize(preset='beginner_weight_loss'):
    """Initialize or reset all components."""
    global env, q_agent, approx_agent, search_agent, vi_agent, current_profile, simulation_log

    current_profile = UserProfile(preset_name=preset)
    env = FitnessEnv(user_profile=current_profile)

    q_agent = QLearningAgent()
    approx_agent = ApproximateQLearningAgent(
        goal_type=current_profile.goal_type,
        target_state=current_profile.target_state
    )
    search_agent = AStarSearchAgent(env)
    vi_agent = ValueIterationAgent(env)
    simulation_log = []


# Initialize on startup
_initialize()


# ═══════════════════════════════════════════════════════════════
# Page Routes
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def dashboard():
    """Serve the main dashboard page."""
    return render_template('dashboard.html')


# ═══════════════════════════════════════════════════════════════
# API Routes
# ═══════════════════════════════════════════════════════════════

@app.route('/api/setup', methods=['POST'])
def setup():
    """
    Configure user profile and reset environment.
    
    Expects JSON: { "preset": "beginner_weight_loss" }
    Or custom:    { "custom": { "name": ..., "goal_type": ..., ... } }
    """
    data = request.get_json() or {}
    preset = data.get('preset', 'beginner_weight_loss')
    custom = data.get('custom', None)

    if custom:
        _initialize.__wrapped__ = None  # placeholder
        global env, q_agent, approx_agent, search_agent, current_profile, simulation_log
        current_profile = UserProfile(custom_config=custom)
        env = FitnessEnv(user_profile=current_profile)
        q_agent = QLearningAgent()
        approx_agent = ApproximateQLearningAgent(
            goal_type=current_profile.goal_type,
            target_state=current_profile.target_state
        )
        search_agent = AStarSearchAgent(env)
        simulation_log = []
    else:
        _initialize(preset)

    return jsonify({
        'status': 'ok',
        'profile': current_profile.to_dict(),
        'state': env.get_state_dict(),
    })


@app.route('/api/train', methods=['POST'])
def train():
    """
    Train Q-Learning and/or Approximate Q-Learning agents.

    Expects JSON: { "episodes": 500, "agent": "both", "resume": false,
                     "hyperparams": {"alpha": 0.1, "gamma": 0.95, ...} }
    Agent options: "q_learning", "approximate", "both"
    resume=true continues training without resetting agents.
    """
    global q_agent, approx_agent

    data = request.get_json() or {}
    episodes = min(data.get('episodes', NUM_EPISODES), 2000)
    agent_type = data.get('agent', 'both')
    resume = data.get('resume', False)
    hp = data.get('hyperparams', {})

    # Apply hyperparams when starting fresh training
    if not resume and hp:
        alpha = hp.get('alpha', 0.1)
        gamma = hp.get('gamma', 0.95)
        eps_start = hp.get('epsilon_start', 1.0)
        eps_end = hp.get('epsilon_end', 0.05)
        q_agent = QLearningAgent(
            learning_rate=alpha, discount_factor=gamma,
            epsilon_start=eps_start, epsilon_end=eps_end
        )
        approx_agent = ApproximateQLearningAgent(
            learning_rate=alpha, discount_factor=gamma,
            epsilon_start=eps_start, epsilon_end=eps_end
        )

    results = {}

    if agent_type in ('q_learning', 'both'):
        train_env = FitnessEnv(user_profile=current_profile)
        if not resume and not hp:
            q_agent.reset_agent()
        start_time = time.time()
        q_metrics = q_agent.train(train_env, num_episodes=episodes)
        q_metrics['training_time'] = round(time.time() - start_time, 2)
        results['q_learning'] = q_metrics

    if agent_type in ('approximate', 'both'):
        train_env = FitnessEnv(user_profile=current_profile)
        if not resume and not hp:
            approx_agent.reset_agent()
        start_time = time.time()
        approx_metrics = approx_agent.train(train_env, num_episodes=episodes)
        approx_metrics['training_time'] = round(time.time() - start_time, 2)
        results['approximate'] = approx_metrics

    return jsonify({'status': 'ok', 'results': results})


@app.route('/api/search', methods=['POST'])
def search():
    """
    Run A* search to find the optimal workout plan.

    Expects JSON: { "max_depth": 14 }
    """
    data = request.get_json() or {}
    max_depth = min(data.get('max_depth', 14), 21)

    # Reset env for search
    search_env = FitnessEnv(user_profile=current_profile)
    agent = AStarSearchAgent(search_env, max_depth=max_depth)

    start_time = time.time()
    result = agent.search()
    result['search_time'] = round(time.time() - start_time, 2)

    return jsonify({'status': 'ok', 'result': result})


@app.route('/api/value_iteration', methods=['POST'])
def value_iteration():
    """
    Run Value Iteration to solve the MDP exactly (Lecture 8).

    Expects JSON: { "iterations": 50 }
    """
    data = request.get_json() or {}
    iterations = min(data.get('iterations', 50), 100)

    vi_env = FitnessEnv(user_profile=current_profile)
    global vi_agent
    vi_agent = ValueIterationAgent(vi_env, num_iterations=iterations)

    result = vi_agent.solve()
    return jsonify({'status': 'ok', 'result': result})


@app.route('/api/recommend', methods=['GET'])
def recommend():
    """
    Get recommendations from all agents for the current state.
    """
    state = env.state
    state_dict = env.get_state_dict()

    recommendations = {'state': state_dict}

    # Q-Learning recommendation
    q_rec = q_agent.get_recommendation(state)
    q_rec['action_description'] = env.get_action_description(q_rec['best_action'])
    recommendations['q_learning'] = q_rec

    # Approximate Q-Learning recommendation
    approx_rec = approx_agent.get_recommendation(state)
    approx_rec['action_description'] = env.get_action_description(approx_rec['best_action'])
    recommendations['approximate'] = approx_rec

    # Value Iteration recommendation
    if vi_agent and vi_agent.policy:
        vi_rec = vi_agent.get_recommendation(state)
        vi_rec['action_description'] = env.get_action_description(vi_rec['best_action'])
        recommendations['value_iteration'] = vi_rec

    return jsonify({'status': 'ok', 'recommendations': recommendations})


@app.route('/api/simulate', methods=['POST'])
def simulate():
    """
    Simulate one step using a specified agent's recommendation.

    Expects JSON: { "agent": "q_learning" }
    Options: "q_learning", "approximate", "manual"
    For manual: { "agent": "manual", "action": 5 }
    """
    data = request.get_json() or {}
    agent_type = data.get('agent', 'q_learning')

    state = env.state

    if agent_type == 'q_learning':
        action = q_agent.get_action(state, explore=False)
    elif agent_type == 'approximate':
        action = approx_agent.get_action(state, explore=False)
    elif agent_type == 'value_iteration' and vi_agent and vi_agent.policy:
        action = vi_agent.get_action(state)
    elif agent_type == 'manual':
        action = data.get('action', 0)
    else:
        action = q_agent.get_action(state, explore=False)

    next_state, reward, done, truncated, info = env.step(action)

    action_desc = env.get_action_description(action)
    log_entry = {
        'step': len(simulation_log) + 1,
        'action': action_desc,
        'reward': round(reward, 2),
        'new_state': env.get_state_dict(),
        'done': done,
        'truncated': truncated,
    }
    simulation_log.append(log_entry)

    return jsonify({
        'status': 'ok',
        'step': log_entry,
        'done': done,
        'truncated': truncated,
        'total_reward': round(env.total_reward, 2),
        'history': env.history,
    })


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset the environment (keeps trained agents)."""
    global simulation_log
    env.reset()
    simulation_log = []
    return jsonify({
        'status': 'ok',
        'state': env.get_state_dict(),
    })


@app.route('/api/metrics', methods=['GET'])
def metrics():
    """Get all current metrics and agent status."""
    return jsonify({
        'status': 'ok',
        'q_learning': q_agent.get_training_metrics(),
        'approximate': approx_agent.get_training_metrics(),
        'state': env.get_state_dict(),
        'simulation_log': simulation_log,
        'profile': current_profile.to_dict(),
    })


@app.route('/api/presets', methods=['GET'])
def presets():
    """List available user profile presets."""
    preset_list = []
    for key, val in UserProfile.PRESETS.items():
        preset_list.append({
            'key': key,
            'name': val['name'],
            'goal_type': val['goal_type'],
        })
    return jsonify({'status': 'ok', 'presets': preset_list})


@app.route('/api/actions', methods=['GET'])
def actions():
    """List all available actions with descriptions."""
    from config import NUM_ACTIONS
    action_list = []
    for a in range(NUM_ACTIONS):
        desc = env.get_action_description(a)
        action_list.append({'index': a, **desc})
    return jsonify({'status': 'ok', 'actions': action_list})


# ═══════════════════════════════════════════════════════════════
# Run Server
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  🏋️  AI Health & Fitness Recommendation System")
    print("  📊 Dashboard: http://{}:{}".format(WEB_HOST, WEB_PORT))
    print("=" * 60 + "\n")
    app.run(host=WEB_HOST, port=WEB_PORT, debug=DEBUG)
