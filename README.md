# Gridworld Q-Learning Simulation

An interactive gridworld environment for experimenting with tabular Q-learning reinforcement learning. Features a complete GUI for grid design, parameter tuning, training visualization, and policy analysis.

## Features

- **Interactive Grid Builder**: Design custom gridworlds with walls, start, and goal positions
- **Real-time Training Progress**: Visual progress bar showing training episodes
- **Comprehensive Visualizations**:
  - Value heatmaps showing learned state values
  - Reward heatmaps for grid structure
  - Full policy arrows for all states
  - Optimal path visualization
  - Episode comparison animations
- **Performance Metrics**: Evaluation against BFS optimal path with success rates
- **Modular Architecture**: Separated environment, agent, and UI components
- **Full Test Coverage**: Unit tests for core functionality
- **CI/CD Ready**: GitHub Actions workflow included

## Project Structure

```
temporal_difference_learning/
├── env.py                  # GridEnvironment class (transitions, rewards, BFS)
├── agent.py                # RLAgent class (Q-learning algorithm)
├── ui.py                   # GUI components (grid builder, visualizations)
├── Gridworld_Qlearning.py  # Main entry point
├── tests/                  # Unit tests
│   ├── test_gridworld.py
│   └── test_more.py
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Package metadata
├── .github/workflows/     # CI configuration
└── README.md
```

## Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd temporal_difference_learning
   ```

2. **Create a virtual environment (recommended)**
   
   **Windows (PowerShell):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   
   **Linux/macOS:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Simulation

```bash
python Gridworld_Qlearning.py
```

### Workflow

1. **Parameter Panel**: Set training hyperparameters
   - Number of episodes
   - Max steps per episode
   - Learning rate (alpha)
   - Discount factor (gamma)
   - Exploration rate (epsilon)
   - Evaluation trials

2. **Grid Builder**: Design your gridworld
   - Click tool buttons (Empty, Wall, Start, Goal)
   - Click/drag on canvas to paint cells
   - Adjust grid size if needed
   - Click "Done" when ready

3. **Training**: Watch progress bar as agent learns
   - Real-time episode counter
   - Automatic completion

4. **Results**: View comprehensive analysis
   - Evaluation popup with success metrics
   - Value and reward heatmaps
   - Policy visualization
   - Episode comparison animations

### Example Grids

The simulation includes a default 5×5 grid with walls. You can create custom grids with:
- **Empty cells** (' '): Traversable spaces
- **Walls** ('W'): Impassable barriers
- **Start** ('S'): Agent starting position (green)
- **Goal** ('G'): Target position (red)

## Algorithm Details

### Q-Learning Implementation

- **Update Rule**: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
- **Exploration**: Epsilon-greedy with linear decay
- **Reward Shaping**:
  - +1.0 for reaching goal
  - -0.1 for hitting walls/boundaries
  - -0.01 per step (encourages shorter paths)

### Evaluation

- Greedy policy rollouts (ε=0)
- Comparison against BFS optimal path
- Success rate and path length statistics

## Testing

Run all unit tests:

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = "${PWD}"
pytest -v
```

**Linux/macOS:**
```bash
PYTHONPATH=$PWD pytest -v
```

### Test Coverage

- Environment transitions and boundary handling
- BFS shortest path computation
- Reward shaping mechanics
- Agent training with reproducible seeds
- Policy evaluation

## Development

### Code Organization

- **`env.py`**: Environment dynamics, transition function, reward calculation, BFS pathfinding
- **`agent.py`**: Q-learning algorithm, training loop, policy extraction, evaluation
- **`ui.py`**: Tkinter GUI components, Matplotlib visualizations, modal windows
- **`Gridworld_Qlearning.py`**: Application orchestration and workflow

### Adding Features

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Continuous Integration

GitHub Actions automatically runs tests on:
- Push to main/master branches
- Pull requests

See `.github/workflows/python-app.yml` for configuration.

## Dependencies

- **numpy**: Numerical operations and Q-table storage
- **matplotlib**: Visualization and plotting
- **pytest**: Unit testing framework

## Known Issues

- Large grids (>20×20) may slow down visualization
- Maximized windows may not work on all platforms


---

**Happy Learning! 🎓🤖**
