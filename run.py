"""
AI for Personalized Health & Fitness Recommendations
====================================================
Main entry point — launches the web dashboard.

Usage:
    python run.py              # Start web dashboard
    python run.py --cli        # Run CLI demo (no web)
"""

import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_cli_demo():
    """Run a quick CLI demonstration of all AI agents."""
    from environment.fitness_env import FitnessEnv
    from environment.user_profile import UserProfile
    from agents.q_learning_agent import QLearningAgent
    from agents.approx_q_agent import ApproximateQLearningAgent
    from agents.search_agent import AStarSearchAgent
    from config import MAX_SEARCH_DEPTH

    print("\n" + "=" * 60)
    print("  🏋️  AI Health & Fitness — CLI Demo")
    print("=" * 60)

    # ── Setup ──
    profile = UserProfile(preset_name='beginner_weight_loss')
    env = FitnessEnv(user_profile=profile)
    print(f"\n👤 User: {profile.name}")
    print(f"🎯 Goal: {profile.goal_type}")
    print(f"📊 Initial State: {env.get_state_dict()}")

    # ── Q-Learning Training ──
    print("\n" + "-" * 40)
    print("🧠 Training Q-Learning Agent (500 episodes)...")
    q_agent = QLearningAgent()
    q_metrics = q_agent.train(env, num_episodes=500)
    print(f"  ✅ Avg Reward (last 100): {q_metrics['avg_reward_last_100']}")
    print(f"  ✅ Best Reward: {q_metrics['best_reward']}")
    print(f"  ✅ States Explored: {q_metrics['states_explored']}")

    # ── Approximate Q-Learning Training ──
    print("\n" + "-" * 40)
    print("⚡ Training Approximate Q-Learning Agent (500 episodes)...")
    approx_agent = ApproximateQLearningAgent(
        goal_type=profile.goal_type,
        target_state=profile.target_state
    )
    approx_env = FitnessEnv(user_profile=profile)
    approx_metrics = approx_agent.train(approx_env, num_episodes=500)
    print(f"  ✅ Avg Reward (last 100): {approx_metrics['avg_reward_last_100']}")
    print(f"  ✅ Best Reward: {approx_metrics['best_reward']}")
    print(f"  ✅ Learned Weights: {approx_metrics['weights']}")

    # ── A* Search ──
    print("\n" + "-" * 40)
    print(f"🔍 Running A* Search (depth={MAX_SEARCH_DEPTH})...")
    search_env = FitnessEnv(user_profile=profile)
    search_agent = AStarSearchAgent(search_env, max_depth=MAX_SEARCH_DEPTH)
    search_result = search_agent.search()
    print(f"  ✅ Status: {search_result['status']}")
    print(f"  ✅ Plan Length: {search_result['plan_length']} days")
    print(f"  ✅ Estimated Reward: {search_result['total_estimated_reward']}")
    print(f"  ✅ Nodes Expanded: {search_result['nodes_expanded']}")
    print("\n  📋 Optimal Plan:")
    for day in search_result['plan']:
        print(f"    Day {day['day']}: {day['description']}")

    # ── Simulate with Q-Learning ──
    print("\n" + "-" * 40)
    print("▶️ Simulating 30 days with Q-Learning agent...")
    sim_env = FitnessEnv(user_profile=profile)
    state = sim_env.reset()
    for step in range(30):
        action = q_agent.get_action(state, explore=False)
        state, reward, done, truncated, info = sim_env.step(action)
        desc = sim_env.get_action_description(action)
        print(f"  Day {step+1}: {desc['description']} → Reward: {reward:+.2f}")
        if done:
            print("  🎉 Goal Reached!")
            break

    print(f"\n📊 Final State: {sim_env.get_state_dict()}")
    print(f"💰 Total Reward: {sim_env.total_reward:.2f}")
    print("\n" + "=" * 60 + "\n")


def run_web():
    """Launch the Flask web dashboard."""
    from web.app import app
    from config import WEB_HOST, WEB_PORT, DEBUG

    print("\n" + "=" * 60)
    print("  🏋️  AI Health & Fitness Recommendation System")
    print(f"  📊 Dashboard: http://{WEB_HOST}:{WEB_PORT}")
    print("=" * 60 + "\n")

    app.run(host=WEB_HOST, port=WEB_PORT, debug=DEBUG)


if __name__ == '__main__':
    if '--cli' in sys.argv:
        run_cli_demo()
    else:
        run_web()
