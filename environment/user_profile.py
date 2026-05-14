"""
User Profile — defines user characteristics and fitness goals.

This module manages the simulated user's physical attributes,
current fitness state, and target goals for the RL system.
"""


class UserProfile:
    """
    Represents a user with physical attributes and a fitness goal.
    
    The profile defines the starting state for the MDP environment
    and the goal that shapes the reward function.
    """

    # Default profiles for quick testing
    PRESETS = {
        'beginner_weight_loss': {
            'name': 'Ahmed (Beginner)',
            'age': 28,
            'height_cm': 175,
            'weight_kg': 92,
            'goal_type': 'weight_loss',
            'target_weight_kg': 78,
            'initial_state': {
                'fitness': 2,
                'energy': 3,
                'weight': 3,       # Overweight
                'muscle': 1,
                'fatigue': 0,
            },
            'target_state': {
                'fitness': 6,
                'energy': 3,
                'weight': 2,       # Normal
                'muscle': 2,
                'fatigue': 0,
            },
        },
        'intermediate_muscle': {
            'name': 'Khaled (Intermediate)',
            'age': 24,
            'height_cm': 180,
            'weight_kg': 75,
            'goal_type': 'muscle_gain',
            'target_weight_kg': 82,
            'initial_state': {
                'fitness': 5,
                'energy': 3,
                'weight': 2,       # Normal
                'muscle': 2,
                'fatigue': 0,
            },
            'target_state': {
                'fitness': 7,
                'energy': 3,
                'weight': 2,
                'muscle': 4,       # Very strong
                'fatigue': 0,
            },
        },
        'advanced_fitness': {
            'name': 'Omar (Advanced)',
            'age': 30,
            'height_cm': 170,
            'weight_kg': 68,
            'goal_type': 'general_fitness',
            'target_weight_kg': 68,
            'initial_state': {
                'fitness': 6,
                'energy': 3,
                'weight': 2,
                'muscle': 3,
                'fatigue': 1,
            },
            'target_state': {
                'fitness': 9,
                'energy': 4,
                'weight': 2,
                'muscle': 4,
                'fatigue': 0,
            },
        },
    }

    def __init__(self, preset_name='beginner_weight_loss', custom_config=None):
        """
        Initialize a user profile from a preset or custom config.

        Args:
            preset_name: Key into PRESETS dict
            custom_config: Optional dict to override preset values
        """
        if custom_config:
            config = custom_config
        else:
            config = self.PRESETS.get(preset_name, self.PRESETS['beginner_weight_loss'])

        self.name = config.get('name', 'User')
        self.age = config.get('age', 25)
        self.height_cm = config.get('height_cm', 175)
        self.weight_kg = config.get('weight_kg', 80)
        self.goal_type = config.get('goal_type', 'general_fitness')
        self.target_weight_kg = config.get('target_weight_kg', 75)
        self.equipment = config.get('equipment', 'gym')
        self.time_available = config.get('time_available', 60)
        self.initial_state = config.get('initial_state', {
            'fitness': 3, 'energy': 3, 'weight': 2,
            'muscle': 2, 'fatigue': 0,
        })
        self.target_state = config.get('target_state', None)

        # Auto-generate target_state if not provided
        if not self.target_state:
            init = self.initial_state
            if self.goal_type == 'weight_loss':
                self.target_state = {
                    'fitness': min(9, init.get('fitness', 3) + 4),
                    'energy': 3, 'weight': max(0, init.get('weight', 3) - 2),
                    'muscle': min(4, init.get('muscle', 2) + 1), 'fatigue': 0,
                }
            elif self.goal_type == 'muscle_gain':
                self.target_state = {
                    'fitness': min(9, init.get('fitness', 3) + 2),
                    'energy': 3, 'weight': init.get('weight', 2),
                    'muscle': 4, 'fatigue': 0,
                }
            elif self.goal_type == 'endurance':
                self.target_state = {
                    'fitness': 9, 'energy': 4,
                    'weight': init.get('weight', 2),
                    'muscle': min(4, init.get('muscle', 2) + 1), 'fatigue': 0,
                }
            else:  # general_fitness
                self.target_state = {
                    'fitness': min(9, init.get('fitness', 3) + 3),
                    'energy': 4, 'weight': 2,
                    'muscle': min(4, init.get('muscle', 2) + 2), 'fatigue': 0,
                }

    def to_dict(self):
        """Serialize profile to a JSON-friendly dictionary."""
        return {
            'name': self.name,
            'age': self.age,
            'height_cm': self.height_cm,
            'weight_kg': self.weight_kg,
            'goal_type': self.goal_type,
            'target_weight_kg': self.target_weight_kg,
            'initial_state': self.initial_state,
            'target_state': self.target_state,
        }

    @staticmethod
    def list_presets():
        """Return available preset names."""
        return list(UserProfile.PRESETS.keys())
