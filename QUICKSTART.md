# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/Pogaldock/Gridworld-Q_Learning-TD-Simulation.git
cd Gridworld-Q_Learning-TD-Simulation

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# Or (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Run the Simulation

```bash
python main.py
```

## Workflow

### 1. Set Parameters
Configure training hyperparameters:
- **Episodes**: Number of training episodes (e.g., 5000)
- **Max Steps**: Maximum steps per episode (e.g., 100)
- **Alpha (α)**: Learning rate (e.g., 0.01)
- **Gamma (γ)**: Discount factor (e.g., 0.95)
- **Epsilon (ε)**: Initial exploration rate (e.g., 0.1)

### 2. Build Grid
Design your gridworld:
- Select **Wall** tool and draw obstacles
- Place one **Start (S)** position (green)
- Place one **Goal (G)** position (red)
- Optionally use **Maze Generation** buttons for automatic mazes
- Click **Done**

### 3. Training
Watch the progress bar as the agent learns through Q-learning.

### 4. Results
View:
- **Animation Viewer**: Step through episodes, see agent's learning progress
- **Heatmaps**: Value function and rewards visualization
- **Policy Arrows**: Learned optimal directions for each state
- **Statistics**: Success rate, path length comparison with optimal (BFS)

## Quick Example

1. Run `python main.py`
2. Use default parameters (click Run)
3. Click "Classic Maze" button in grid builder
4. Click "Done"
5. Wait for training (~5 seconds)
6. Explore the results!

## Keyboard Shortcuts in Animation Viewer

- **First**: Jump to first episode
- **Previous**: Previous episode
- **Play/Pause**: Toggle animation
- **Next**: Next episode
- **Last**: Jump to last episode
- **First Success**: Jump to first successful episode
- **First Optimal**: Jump to first episode with optimal path

## Tips

- **Small grids** (5×5 to 10×10): Use 1000-5000 episodes
- **Large grids** (15×15+): Use 10000+ episodes
- **Complex mazes**: Increase max_steps and episodes
- **Faster training**: Increase alpha (but may be less stable)
- **Better exploration**: Start with higher epsilon

## Troubleshooting

**Grid builder not showing?**
- Check that tkinter is installed: `python -m tkinter`

**Training too slow?**
- Reduce number of episodes
- Use smaller grid
- Close other applications

**Agent not reaching goal?**
- Increase episodes or max_steps
- Check maze has valid path (use BFS path length indicator)
- Adjust learning parameters (try alpha=0.1)

Enjoy experimenting with Q-Learning! 🎓🤖
