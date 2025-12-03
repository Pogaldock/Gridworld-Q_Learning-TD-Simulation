"""Configuration constants for the Gridworld Q-Learning simulation."""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOUSE_ICON_PATH = os.path.join(BASE_DIR, "assets", "mouse_icon.png")

# Grid colors
GRID_COLORS = {
    " ": "white",
    "W": "gray",
    "S": "green",
    "G": "red"
}

# Default parameters
DEFAULT_PARAMS = {
    "episodes": 10000,
    "max_steps": 200,
    "epsilon": 0.1,
    "epsilon_end": 0.01,
    "alpha": 0.01,
    "gamma": 0.95,
    "seed": 0,
    "eval_trials": 50
}

# UI settings
MIN_CELL_SIZE = 10
DEFAULT_CELL_SIZE = 40
MIN_GRID_SIZE = 2
MAX_GRID_SIZE = 100

# Animation speeds
ANIMATION_SPEEDS = {
    "Slow": 150,
    "Normal": 50,
    "Fast": 20,
    "Very Fast": 5
}

# Direction angles for mouse rotation
DIRECTION_ANGLES = {
    "up": 0,
    "down": 180,
    "left": 270,
    "right": 90
}

# Progress bar settings
PROGRESS_BAR_LENGTH = 40
PROGRESS_UPDATE_INTERVAL = 0.5  # seconds
