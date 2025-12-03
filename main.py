#!/usr/bin/env python
"""Main entry point for Gridworld Q-Learning Simulation."""
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from Gridworld_Qlearning import main

if __name__ == "__main__":
    main()
