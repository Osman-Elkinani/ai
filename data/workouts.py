"""
Workout Database — defines all available workout types and their effects.

Each workout specifies:
  - intensity: 0-10 difficulty scale
  - calories_burned: approximate calories per session
  - duration_minutes: session length
  - effects: expected changes to state variables (mean values)
  - requirements: minimum state levels to safely perform
  - description: human-readable explanation
"""

WORKOUT_DATABASE = {
    'rest': {
        'name': 'Rest Day',
        'emoji': '😴',
        'intensity': 0,
        'calories_burned': 0,
        'duration_minutes': 0,
        'effects': {
            'fitness': 0,
            'energy': 2,        # Major recovery
            'weight': 0,
            'muscle': 0,
            'fatigue': -2,      # Fatigue drops significantly
        },
        'requirements': {},
        'description': 'Complete rest for physical and mental recovery.'
    },
    'walking': {
        'name': 'Walking',
        'emoji': '🚶',
        'intensity': 2,
        'calories_burned': 150,
        'duration_minutes': 30,
        'effects': {
            'fitness': 0.3,
            'energy': 0,
            'weight': -0.2,
            'muscle': 0,
            'fatigue': 0.5,
        },
        'requirements': {},
        'description': 'Light 30-minute walk, suitable for all fitness levels.'
    },
    'jogging': {
        'name': 'Jogging',
        'emoji': '🏃',
        'intensity': 5,
        'calories_burned': 350,
        'duration_minutes': 30,
        'effects': {
            'fitness': 0.7,
            'energy': -1,
            'weight': -0.5,
            'muscle': 0.1,
            'fatigue': 1.0,
        },
        'requirements': {'fitness': 2, 'energy': 1},
        'description': 'Moderate 30-minute jog to build cardiovascular endurance.'
    },
    'hiit': {
        'name': 'HIIT Training',
        'emoji': '🔥',
        'intensity': 9,
        'calories_burned': 500,
        'duration_minutes': 25,
        'effects': {
            'fitness': 1.2,
            'energy': -2,
            'weight': -0.8,
            'muscle': 0.3,
            'fatigue': 2.0,
        },
        'requirements': {'fitness': 4, 'energy': 2},
        'description': 'High-intensity interval training for maximum calorie burn.'
    },
    'strength_training': {
        'name': 'Strength Training',
        'emoji': '🏋️',
        'intensity': 7,
        'calories_burned': 300,
        'duration_minutes': 45,
        'effects': {
            'fitness': 0.4,
            'energy': -1,
            'weight': 0.1,      # Slight gain from muscle
            'muscle': 1.0,      # Major muscle growth
            'fatigue': 1.5,
        },
        'requirements': {'energy': 1},
        'description': 'Resistance training to build muscle mass and strength.'
    },
    'swimming': {
        'name': 'Swimming',
        'emoji': '🏊',
        'intensity': 6,
        'calories_burned': 400,
        'duration_minutes': 40,
        'effects': {
            'fitness': 0.8,
            'energy': -1,
            'weight': -0.6,
            'muscle': 0.4,
            'fatigue': 1.0,
        },
        'requirements': {'fitness': 2, 'energy': 1},
        'description': 'Full-body low-impact workout in the pool.'
    },
    'cycling': {
        'name': 'Cycling',
        'emoji': '🚴',
        'intensity': 6,
        'calories_burned': 380,
        'duration_minutes': 40,
        'effects': {
            'fitness': 0.7,
            'energy': -1,
            'weight': -0.5,
            'muscle': 0.3,
            'fatigue': 1.0,
        },
        'requirements': {'fitness': 2, 'energy': 1},
        'description': 'Endurance cycling session for legs and cardio.'
    },
}


def get_workout_info(workout_key):
    """Return the full info dict for a workout type."""
    return WORKOUT_DATABASE.get(workout_key, None)


def get_workout_names():
    """Return a list of (key, display_name) tuples."""
    return [(k, v['name']) for k, v in WORKOUT_DATABASE.items()]
