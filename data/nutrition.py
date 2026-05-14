"""
Nutrition Plan Database — defines all available nutrition strategies.

Each plan specifies:
  - name: Display name
  - daily_calories: approximate caloric target
  - macros: protein/carbs/fat ratios
  - effects: how this nutrition plan affects state variables
  - description: human-readable explanation
"""

NUTRITION_DATABASE = {
    'caloric_deficit': {
        'name': 'Caloric Deficit',
        'emoji': '📉',
        'daily_calories': 1600,
        'macros': {'protein': 0.35, 'carbs': 0.40, 'fat': 0.25},
        'effects': {
            'energy': -0.5,     # Less energy from fewer calories
            'weight': -0.5,     # Weight loss from deficit
            'muscle': -0.1,     # Slight muscle loss risk
            'fatigue': 0.3,     # Slightly more fatigued
        },
        'description': 'Eat below maintenance calories to promote fat loss.'
    },
    'maintenance': {
        'name': 'Maintenance',
        'emoji': '⚖️',
        'daily_calories': 2200,
        'macros': {'protein': 0.30, 'carbs': 0.45, 'fat': 0.25},
        'effects': {
            'energy': 0.5,
            'weight': 0,
            'muscle': 0,
            'fatigue': -0.2,
        },
        'description': 'Eat at maintenance level to sustain current weight.'
    },
    'caloric_surplus': {
        'name': 'Caloric Surplus',
        'emoji': '📈',
        'daily_calories': 2800,
        'macros': {'protein': 0.30, 'carbs': 0.50, 'fat': 0.20},
        'effects': {
            'energy': 1.0,      # More energy from extra calories
            'weight': 0.5,      # Weight gain
            'muscle': 0.3,      # Better muscle building
            'fatigue': -0.5,    # Better recovery
        },
        'description': 'Eat above maintenance to support muscle growth.'
    },
    'high_protein': {
        'name': 'High Protein',
        'emoji': '🥩',
        'daily_calories': 2200,
        'macros': {'protein': 0.45, 'carbs': 0.35, 'fat': 0.20},
        'effects': {
            'energy': 0.3,
            'weight': -0.1,
            'muscle': 0.5,      # Best for muscle preservation/growth
            'fatigue': -0.3,    # Good recovery
        },
        'description': 'High protein intake to maximize muscle synthesis.'
    },
}


def get_nutrition_info(plan_key):
    """Return the full info dict for a nutrition plan."""
    return NUTRITION_DATABASE.get(plan_key, None)


def get_nutrition_names():
    """Return a list of (key, display_name) tuples."""
    return [(k, v['name']) for k, v in NUTRITION_DATABASE.items()]
