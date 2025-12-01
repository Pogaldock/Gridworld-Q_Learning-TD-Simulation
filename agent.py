import numpy as np
import statistics
from typing import Optional, Tuple, List, Dict
from env import GridEnvironment


class RLAgent:
    """Tabular Q-learning agent for GridEnvironment.

    Provides training, greedy rollout evaluation and policy extraction.
    """

    def __init__(self, env: GridEnvironment, alpha: float = 0.1, gamma: float = 0.95,
                 epsilon_start: float = 0.1, epsilon_end: float = 0.01,
                 max_steps: int = 100, seed: Optional[int] = None):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.max_steps = max_steps
        self.actions = list(GridEnvironment.ACTIONS)
        self.q: Dict[Tuple[Tuple[int, int], str], float] = {}
        self.seed = seed

    def init_q(self, init_value: float = 0.0):
        rows, cols = self.env.rows, self.env.cols
        states = [(r, c) for r in range(rows) for c in range(cols) if self.env.grid[r, c] != 'W']
        self.q = {(s, a): init_value for s in states for a in self.actions}
        return states

    def choose_action(self, state: Tuple[int, int], epsilon: float) -> str:
        if np.random.rand() < epsilon:
            return np.random.choice(self.actions)
        qs = [self.q[(state, a)] for a in self.actions]
        max_q = max(qs)
        best = [a for a, q in zip(self.actions, qs) if q == max_q]
        return np.random.choice(best)

    def update_q(self, state, action, next_state):
        r = self.env.reward(state, next_state)
        if self.env.grid[next_state] == 'G':
            target = r
        else:
            target = r + self.gamma * max(self.q[(next_state, a)] for a in self.actions)
        self.q[(state, action)] += self.alpha * (target - self.q[(state, action)])

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

    def train(self, episodes: int = 1000, record_eps: Optional[List[int]] = None, progress_callback=None):
        if self.seed is not None:
            np.random.seed(self.seed)

        states = self.init_q()
        start = self.env.start
        if record_eps is None:
            record_eps = [1, 2, 3, 4, 5, max(1, episodes - 1)]
        record_indices = [max(0, e - 1) for e in record_eps]
        recorded_paths: Dict[int, List[Tuple[int, int]]] = {}

        for ep in range(episodes):
            # linear decay
            if episodes > 1:
                eps = max(self.epsilon_end, self.epsilon_start + (self.epsilon_end - self.epsilon_start) * (ep / (episodes - 1)))
            else:
                eps = self.epsilon_start
            p = self.run_episode(start, eps)
            if ep in record_indices:
                recorded_paths[ep + 1] = p
            # notify progress (ep number is 0-based; report 1-based count)
            try:
                if progress_callback is not None:
                    progress_callback(ep + 1, episodes)
            except Exception:
                # don't fail training if UI callback misbehaves
                pass

        return self.q, self.actions, states, recorded_paths

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

    def evaluate(self, trials: int = 50) -> dict:
        """Run `trials` greedy rollouts (epsilon=0) and return statistics.

        Returns a dict with keys: 'success_count', 'success_rate', 'lengths', 'min', 'median', 'mean'.
        Failed runs are recorded as None in lengths and ignored for min/mean/median calculations.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        lengths = []
        for _ in range(trials):
            path = self.greedy_rollout(self.env.start)
            if len(path) >= 1 and self.env.grid[path[-1]] == 'G':
                lengths.append(len(path) - 1)
            else:
                lengths.append(None)

        successes = [l for l in lengths if l is not None]
        success_count = len(successes)
        stats = {
            'success_count': success_count,
            'success_rate': success_count / trials if trials else 0.0,
            'lengths': lengths,
            'min': min(successes) if successes else None,
            'median': statistics.median(successes) if successes else None,
            'mean': statistics.mean(successes) if successes else None,
        }
        return stats

    def extract_policy(self):
        rows, cols = self.env.rows, self.env.cols
        P = np.full((rows, cols), ' ', dtype=str)
        for r in range(rows):
            for c in range(cols):
                if self.env.grid[r, c] == 'W':
                    P[r, c] = '#'
                elif self.env.grid[r, c] == 'S':
                    P[r, c] = 'S'
                elif self.env.grid[r, c] == 'G':
                    P[r, c] = 'G'
                else:
                    qvals = {a: self.q[((r, c), a)] for a in self.actions}
                    best = max(qvals, key=qvals.get)
                    P[r, c] = best[0].upper()
        return P

    def trace_policy_path(self, max_steps: int = 500) -> List[Tuple[int, int]]:
        start = self.env.start
        curr = start
        path = [curr]
        visited = {curr}
        for _ in range(max_steps):
            if self.env.grid[curr] == 'G':
                break
            qvals = {a: self.q[(curr, a)] for a in self.actions}
            best = max(qvals, key=qvals.get)
            nxt = self.env.transition(curr, best)
            if nxt == curr or nxt in visited:
                break
            visited.add(nxt)
            path.append(nxt)
            curr = nxt
        return path
