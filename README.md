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

## Theoretical Foundations

This section provides a comprehensive mathematical foundation for understanding the Q-Learning and Temporal Difference (TD) Learning implementation in this gridworld environment. We'll build from first principles, connecting theory to the actual code implementation.

### 1. Introduction to Reinforcement Learning

Reinforcement Learning (RL) is a framework for learning optimal behavior through interaction with an environment. Unlike supervised learning, where we have labeled examples, RL agents learn by trial and error, receiving feedback in the form of rewards.

**The Agent-Environment Framework:**

At each time step *t*, the interaction follows this cycle:
- The **agent** observes the current **state** *s_t*
- The agent selects and executes an **action** *a_t*
- The **environment** transitions to a new state *s_{t+1}*
- The agent receives a **reward** *r_t*

**In our Gridworld context:**
- **States**: Grid positions (row, column), e.g., `(0, 0)`, `(2, 3)`
- **Actions**: Movement directions: `'up'`, `'down'`, `'left'`, `'right'`
- **Rewards**: Numerical feedback based on the transition (more details in Section 7)
- **Policy**: A strategy π that maps states to actions, π(s) = a

The agent's goal is to learn a policy that maximizes the cumulative reward over time. In our case, this means finding the shortest path from start to goal while avoiding walls.

### 2. Markov Decision Processes (MDPs)

Our gridworld is a **Markov Decision Process (MDP)**, a mathematical framework for modeling sequential decision-making problems.

**MDP Components:**

An MDP is defined by the tuple (S, A, P, R, γ):
- **S**: State space - all possible grid positions (excluding walls)
- **A**: Action space - `{'up', 'down', 'left', 'right'}`
- **P**: Transition function - P(s'|s,a) gives probability of reaching state s' from state s via action a
- **R**: Reward function - R(s,a,s') gives reward for transition
- **γ**: Discount factor - weighs immediate vs. future rewards (0 ≤ γ < 1)

**Markov Property:** The future depends only on the current state, not the history:
```
P(s_{t+1} | s_t, a_t, s_{t-1}, ..., s_0) = P(s_{t+1} | s_t, a_t)
```

**Implementation in `env.py`:**

The transition function (lines 34-44) implements deterministic dynamics:

```python
def transition(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
    """Return next state after applying action, respecting walls and bounds."""
    r, c = state
    dr = {'up': -1, 'down': 1, 'left': 0, 'right': 0}[action]
    dc = {'up': 0,  'down': 0, 'left': -1, 'right': 1}[action]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < self.rows and 0 <= nc < self.cols):
        return state
    if self.grid[nr, nc] == 'W':
        return state
    return (nr, nc)
```

The reward function (lines 46-52) defines the incentive structure:

```python
def reward(self, state: Tuple[int, int], next_state: Tuple[int, int]) -> float:
    """Reward shaping: +1 for goal, penalty for bumping, small per-step penalty."""
    if self.grid[next_state] == 'G':
        return 1.0
    if next_state == state:
        return -0.1
    return -0.01
```

### 3. The Bellman Equation

The **Bellman Equation** is the foundation of dynamic programming and reinforcement learning. It expresses the value of a state recursively in terms of immediate reward and future value.

**The Optimal Q-Value Function:**

The Q-value Q*(s,a) represents the expected cumulative reward when taking action *a* in state *s* and following the optimal policy thereafter:

```
Q*(s,a) = E[r + γ max_{a'} Q*(s', a')]
```

**Breaking down each term:**
- **Q*(s,a)**: The optimal value of taking action *a* in state *s*
- **r**: Immediate reward received for the transition
- **γ**: Discount factor (0 ≤ γ < 1) - controls how much we value future rewards
  - γ = 0: Only immediate rewards matter (myopic)
  - γ → 1: Future rewards are valued nearly as much as immediate rewards
- **max_{a'} Q*(s', a')**: The value of the best action in the next state *s'*
- **E[...]**: Expectation over possible next states (deterministic in our case)

**Intuition:** The value of a state-action pair equals the immediate reward plus the discounted value of the best future action. This recursive relationship allows us to propagate value information backward through the state space.

**Optimal Policy:** Once we know Q*(s,a) for all state-action pairs, the optimal policy is:
```
π*(s) = argmax_a Q*(s,a)
```

### 4. Temporal Difference Learning

**Temporal Difference (TD) Learning** is a powerful class of RL methods that learn directly from experience without requiring a model of the environment.

**The TD Error:**

The core of TD learning is the **TD error**, which measures the difference between our current estimate and a better estimate:

```
δ = r + γ max_{a'} Q(s', a') - Q(s,a)
```

Where:
- **r + γ max_{a'} Q(s', a')**: The **TD target** - our improved estimate based on actual reward and next state
- **Q(s,a)**: Our current estimate
- **δ**: The error between them

**Why TD Learning is Powerful:**

1. **Bootstrapping**: TD methods learn from their own estimates. We update Q(s,a) using Q(s',a'), which is itself an estimate. This means we can learn before knowing the final outcome.

2. **Online Learning**: We can update after every step, not just at episode end.

3. **No Model Required**: We don't need to know the transition probabilities P(s'|s,a) or reward function R(s,a,s') in advance - we learn from experience.

**Comparison with Monte Carlo:**
- **Monte Carlo**: Wait until episode ends, observe total return, then update
  - Pro: Unbiased estimates
  - Con: High variance, slow learning, requires episodic tasks
- **TD Learning**: Update immediately after each step using bootstrapping
  - Pro: Lower variance, faster learning, works for continuing tasks
  - Con: Biased estimates initially (but converges to truth)

### 5. Q-Learning Algorithm

**Q-Learning** is an off-policy TD control algorithm that learns the optimal action-value function Q* regardless of the policy being followed.

**The Q-Learning Update Rule:**

```
Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]
```

**Component Breakdown:**

1. **α (alpha)**: Learning rate (0 < α ≤ 1)
   - Controls how much we update toward the new information
   - α = 0: Never learn (keep old values)
   - α = 1: Completely replace old value with new estimate
   - Typical values: 0.1 - 0.5

2. **γ (gamma)**: Discount factor (0 ≤ γ < 1)
   - Determines importance of future rewards
   - In our implementation: γ = 0.95 (values future highly)

3. **TD Target**: r + γ max_{a'} Q(s',a')
   - The "truth" we're learning toward
   - Combines actual reward with estimate of future value

4. **Current Estimate**: Q(s,a)
   - What we currently think the value is

5. **TD Error**: [r + γ max_{a'} Q(s',a') - Q(s,a)]
   - How wrong our estimate was

**Implementation in `agent.py` (lines 40-46):**

```python
def update_q(self, state, action, next_state):
    r = self.env.reward(state, next_state)
    if self.env.grid[next_state] == 'G':
        target = r
    else:
        target = r + self.gamma * max(self.q[(next_state, a)] for a in self.actions)
    self.q[(state, action)] += self.alpha * (target - self.q[(state, action)])
```

**Why Q-Learning is "Off-Policy":**

Q-learning learns the optimal policy (exploiting) while following an exploratory policy (ε-greedy). This separation is powerful:
- The policy we follow (behavior policy) can be exploratory
- The policy we learn (target policy) is always greedy/optimal
- This allows efficient exploration while learning optimal behavior

### 6. Exploration vs. Exploitation: Epsilon-Greedy

The **exploration-exploitation tradeoff** is central to reinforcement learning:
- **Exploitation**: Use current knowledge to maximize reward (choose best known action)
- **Exploration**: Try new actions to discover potentially better options

**Epsilon-Greedy Policy:**

With probability ε, explore (random action); otherwise exploit (best action):

```
π(s) = {
    random action from A     with probability ε
    argmax_a Q(s,a)          with probability 1-ε
}
```

**Mathematical Formulation:**
```
P(a|s) = {
    ε/|A| + (1-ε)           if a = argmax_a' Q(s,a')
    ε/|A|                   otherwise
}
```

**Implementation in `agent.py` (lines 32-38):**

```python
def choose_action(self, state: Tuple[int, int], epsilon: float) -> str:
    if np.random.rand() < epsilon:
        return np.random.choice(self.actions)
    qs = [self.q[(state, a)] for a in self.actions]
    max_q = max(qs)
    best = [a for a, q in zip(self.actions, qs) if q == max_q]
    return np.random.choice(best)
```

**Epsilon Decay Schedule:**

As training progresses, we gradually reduce exploration in favor of exploitation. We use **linear decay**:

```
ε(t) = max(ε_end, ε_start + (ε_end - ε_start) * (t / T))
```

Where:
- **ε_start**: Initial exploration rate (e.g., 0.1)
- **ε_end**: Final exploration rate (e.g., 0.01)
- **t**: Current episode
- **T**: Total episodes

**Implementation in `agent.py` (lines 74-77):**

```python
for ep in range(episodes):
    # linear decay
    if episodes > 1:
        eps = max(self.epsilon_end, self.epsilon_start + (self.epsilon_end - self.epsilon_start) * (ep / (episodes - 1)))
    else:
        eps = self.epsilon_start
```

**Why Decay?**
- **Early training**: High ε ensures broad exploration, discovering all parts of the state space
- **Late training**: Low ε focuses on exploiting learned knowledge, fine-tuning the policy
- **Prevents premature convergence** to suboptimal policies

### 7. Reward Shaping

**Reward shaping** is the art of designing rewards that guide the agent toward desired behavior while enabling efficient learning.

**Our Reward Structure:**

1. **Goal Reward: +1.0**
   - Large positive reward for reaching the goal
   - Provides clear objective signal
   - Terminal reward (episode ends)

2. **Wall Penalty: -0.1**
   - Moderate penalty for invalid moves (hitting walls or boundaries)
   - Discourages the agent from repeatedly trying blocked paths
   - Helps distinguish "stuck" states

3. **Step Penalty: -0.01**
   - Small negative reward for each step taken
   - Encourages finding shorter paths
   - Creates urgency to reach goal quickly
   - Without this, any path to goal would be equally good

**Mathematical Impact:**

The cumulative reward for a path of length *n* steps:
```
G = -0.01 * (n-1) + 1.0 = 1.0 - 0.01(n-1)
```

This means:
- 5-step path: G = 1.0 - 0.04 = 0.96
- 10-step path: G = 1.0 - 0.09 = 0.91
- 20-step path: G = 1.0 - 0.19 = 0.81

Shorter paths get higher returns, naturally guiding the agent toward optimal behavior.

**Implementation in `env.py` (lines 46-52):**

```python
def reward(self, state: Tuple[int, int], next_state: Tuple[int, int]) -> float:
    """Reward shaping: +1 for goal, penalty for bumping, small per-step penalty."""
    if self.grid[next_state] == 'G':
        return 1.0
    if next_state == state:
        return -0.1
    return -0.01
```

**Design Principles:**
- **Sparse rewards** (only goal gives significant positive reward) keep the problem realistic
- **Dense shaping** (per-step penalty) provides learning signal at every step
- **Clear gradients** guide the agent toward better solutions

### 8. The Training Loop

Training consists of many **episodes**, where each episode is a complete journey from start to goal (or max steps).

**Single Episode Flow:**

1. **Initialize**: Start at state s = start_position
2. **Loop** until goal reached or max_steps exceeded:
   - **Select action**: a = ε-greedy(s)
   - **Execute**: s' = transition(s, a)
   - **Observe reward**: r = reward(s, s')
   - **Update Q-value**: Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]
   - **Transition**: s ← s'
3. **Record**: Save path for analysis (optional)

**Implementation in `agent.py` (lines 48-59):**

```python
def run_episode(self, start: Tuple[int, int], epsilon: float) -> List[Tuple[int, int]]:
    state = start
    path = [state]
    for _ in range(self.max_steps):
        action = self.choose_action(state, epsilon)
        next_state = self.env.transition(state, action)
        self.update_q(state, action, next_state)
        path.append(next_state)
        if self.env.grid[next_state] == 'G':
            break
        state = next_state
    return path
```

**How Many Episodes?**

The number of episodes needed depends on:
- **Grid complexity**: Larger grids with more walls need more episodes
- **Learning rate α**: Smaller α requires more episodes to converge
- **Exploration rate ε**: Higher ε slows convergence but improves final quality
- **Initialization**: Starting with Q(s,a) = 0 vs. optimistic initialization

**Typical values:**
- Simple 5×5 grids: 500-1000 episodes
- Complex 10×10+ grids: 5000-10000 episodes

**Convergence signs:**
- Q-values stabilize (small updates)
- Policy becomes consistent
- Evaluation performance plateaus

### 9. Policy Extraction and Evaluation

After training, we extract and evaluate the learned policy.

**Policy Definition:**

A policy π is a mapping from states to actions. The optimal policy derived from Q-values is:

```
π*(s) = argmax_a Q(s,a)
```

For each state, choose the action with the highest Q-value. This is the **greedy policy** with respect to the learned Q-function.

**Greedy Rollout (Evaluation):**

To evaluate the learned policy, we perform rollouts with **ε = 0** (purely greedy, no exploration):

1. Start at initial state
2. At each step, select a = argmax_{a'} Q(s, a')
3. Execute and transition
4. Repeat until goal or max steps

**Implementation in `agent.py` (lines 91-105):**

```python
def greedy_rollout(self, start: Tuple[int, int]) -> List[Tuple[int, int]]:
    state = start
    path = [state]
    for _ in range(self.max_steps):
        qs = [self.q[(state, a)] for a in self.actions]
        best = [a for a, q in zip(self.actions, qs) if q == max(qs)]
        action = np.random.choice(best)
        nxt = self.env.transition(state, action)
        path.append(nxt)
        if self.env.grid[nxt] == 'G':
            break
        if nxt == state:
            break
        state = nxt
    return path
```

**Evaluation Metrics:**

1. **Success Rate**: Percentage of rollouts that reach the goal
2. **Path Length**: Number of steps taken (shorter is better)
3. **Comparison with BFS**: 
   - BFS finds the true optimal path (shortest possible)
   - Learned policy ideally matches or approximates BFS path

**BFS Optimal Path (ground truth, from `env.py` lines 54-75):**

```python
def bfs_shortest_path(self) -> List[Tuple[int, int]]:
    """Return shortest path from S to G (list of states), or [] if no path."""
    if self.start is None or self.goal is None:
        return []
    q = deque([self.start])
    parent = {self.start: None}
    while q:
        cur = q.popleft()
        if cur == self.goal:
            # reconstruct path
            path = []
            node = cur
            while node is not None:
                path.append(node)
                node = parent[node]
            return list(reversed(path))
        for a in self.ACTIONS:
            nxt = self.transition(cur, a)
            if nxt != cur and nxt not in parent:
                parent[nxt] = cur
                q.append(nxt)
    return []
```

**Why Compare with BFS?**
- BFS provides **ground truth** - the provably optimal solution
- If learned policy matches BFS path length, we've found optimal behavior
- Small differences are acceptable (multiple optimal paths may exist)
- Large differences indicate insufficient training or poor hyperparameters

### 10. Mathematical Convergence Properties

Q-learning has strong theoretical guarantees under certain conditions.

**Convergence Theorem:**

Under the following conditions, Q-learning converges to the optimal Q-function Q* with probability 1:

1. **Infinite Exploration**: All state-action pairs (s,a) are visited infinitely often
   ```
   ∑_{t=1}^∞ 𝟙[s_t=s, a_t=a] = ∞  for all (s,a)
   ```

2. **Robbins-Monro Conditions** on learning rate α_t:
   ```
   ∑_{t=1}^∞ α_t = ∞  (learns enough)
   ∑_{t=1}^∞ α_t² < ∞  (decreases to zero)
   ```
   
   Examples:
   - Constant α = 0.1 satisfies first, not second (but works in practice)
   - α_t = 1/t satisfies both (theoretical guarantee)
   - α_t = 1/(1 + visits(s,a)) satisfies both (state-specific)

3. **Bounded Rewards**: |r| < ∞ for all rewards
   - Our rewards: {-0.1, -0.01, 1.0} are clearly bounded

**Why Does Epsilon-Greedy Work?**

Our ε-greedy exploration with ε_end > 0 ensures condition 1:
- Every state-action pair has probability ≥ ε/|A| of being selected
- Over infinite episodes, each (s,a) is visited infinitely often
- Even with decay to ε = 0.01, we maintain sufficient exploration

**Practical Considerations:**

In practice:
- **Finite episodes**: We train for a finite number of episodes (e.g., 1000)
- **Fixed α**: We use constant learning rate α = 0.1
- **Approximate convergence**: Q-values get "close enough" to Q*

**Convergence Speed:**

Factors affecting how quickly we approach Q*:
- **Learning rate α**: Higher α converges faster but oscillates more
- **Discount γ**: Higher γ requires more iterations to propagate values
- **Exploration**: More exploration early speeds up discovery
- **Grid complexity**: More states/obstacles slow convergence
- **Reward structure**: Dense rewards provide more learning signal

**Verification:**

We can check convergence by monitoring:
- Q-value stability (small updates indicate convergence)
- Policy consistency (same actions chosen in same states)
- Evaluation performance (success rate and path optimality plateau)

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
