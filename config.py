"""
Configuration constants for the AI Health & Fitness Recommendation System.
All hyperparameters and environment settings are centralized here.
"""

# ═══════════════════════════════════════════════════════════════
# Environment State Space Configuration (MDP - Lectures 7 & 8)
# ═══════════════════════════════════════════════════════════════
FITNESS_LEVELS = 10       # 0-9: cardiovascular fitness
ENERGY_LEVELS = 5         # 0-4: very_low to very_high
WEIGHT_STATUSES = 5       # 0-4: very_underweight to obese
MUSCLE_LEVELS = 5          # 0-4: very_weak to very_strong
FATIGUE_LEVELS = 5         # 0-4: none to severe
DAYS_PER_WEEK = 7          # 0-6: Monday to Sunday

# Total state space size: 10 × 5 × 5 × 5 × 5 × 7 = 43,750
STATE_SPACE_SIZE = (FITNESS_LEVELS * ENERGY_LEVELS * WEIGHT_STATUSES *
                    MUSCLE_LEVELS * FATIGUE_LEVELS * DAYS_PER_WEEK)

# ═══════════════════════════════════════════════════════════════
# Action Space Configuration
# ═══════════════════════════════════════════════════════════════
WORKOUT_TYPES = [
    'rest', 'walking', 'jogging', 'hiit',
    'strength_training', 'swimming', 'cycling'
]

NUTRITION_PLANS = [
    'caloric_deficit', 'maintenance', 'caloric_surplus', 'high_protein'
]

NUM_WORKOUTS = len(WORKOUT_TYPES)       # 8
NUM_NUTRITION = len(NUTRITION_PLANS)    # 4
NUM_ACTIONS = NUM_WORKOUTS * NUM_NUTRITION  # 28

# ═══════════════════════════════════════════════════════════════
# Q-Learning Hyperparameters (Lecture 9: RL I)
# ═══════════════════════════════════════════════════════════════
LEARNING_RATE = 0.1           # α: step size for Q-value updates
DISCOUNT_FACTOR = 0.95        # γ: importance of future rewards
EPSILON_START = 1.0           # Initial exploration rate
EPSILON_END = 0.05            # Minimum exploration rate
EPSILON_DECAY = 0.995         # Multiplicative decay per episode
NUM_EPISODES = 3000           # Training episodes
MAX_STEPS_PER_EPISODE = 30    # ~30 days of simulation per episode

# ═══════════════════════════════════════════════════════════════
# Approximate Q-Learning Hyperparameters (Lecture 10: RL II)
# ═══════════════════════════════════════════════════════════════
APPROX_LEARNING_RATE = 0.01   # Smaller LR for weight updates
NUM_FEATURES = 12             # Number of feature functions

# ═══════════════════════════════════════════════════════════════
# A* Search Configuration (Lecture 3: Informed Search)
# ═══════════════════════════════════════════════════════════════
MAX_SEARCH_DEPTH = 14         # Plan up to 14 days ahead
SEARCH_BEAM_WIDTH = 500       # Max nodes to explore

# ═══════════════════════════════════════════════════════════════
# User Goal Types
# ═══════════════════════════════════════════════════════════════
GOAL_TYPES = ['weight_loss', 'muscle_gain', 'general_fitness', 'endurance']

# ═══════════════════════════════════════════════════════════════
# Web Server Configuration
# ═══════════════════════════════════════════════════════════════
WEB_HOST = '127.0.0.1'
WEB_PORT = 5000
DEBUG = True
