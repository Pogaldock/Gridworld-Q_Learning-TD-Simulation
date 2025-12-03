import numpy as np
from collections import deque
from typing import Optional, Tuple, List


class GridEnvironment:
    """Encapsulates the gridworld: transition dynamics, reward, and utilities.

    Grid cells: ' ' empty, 'W' wall, 'S' start, 'G' goal
    Actions: 'up', 'down', 'left', 'right'
    """

    ACTIONS = ("up", "down", "left", "right")

    def __init__(self, grid: np.ndarray):
        self.grid = np.array(grid, dtype=str)
        self.rows, self.cols = self.grid.shape
        self.start = self._find_one('S')
        self.goal = self._find_one('G')

    def _find_one(self, symbol: str) -> Optional[Tuple[int, int]]:
        locs = np.argwhere(self.grid == symbol)
        if len(locs) == 0:
            return None
        return tuple(locs[0])

    def validate(self) -> None:
        """Raise ValueError if Start or Goal are missing."""
        if self.start is None:
            raise ValueError("Grid has no start cell 'S'")
        if self.goal is None:
            raise ValueError("Grid has no goal cell 'G'")

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

    def reward(self, state: Tuple[int, int], next_state: Tuple[int, int]) -> float:
        """Reward shaping: +1 for goal, penalty for bumping, small per-step penalty."""
        if self.grid[next_state] == 'G':
            return 1.0
        if next_state == state:
            return -0.1
        return -0.01

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


def load_default_grid() -> np.ndarray:
    """A small default grid useful for quick runs and tests."""
    return np.array([
        ['S', 'W', ' ', ' ', ' '],
        [' ', 'W', ' ', 'W', ' '],
        [' ', 'W', ' ', ' ', ' '],
        [' ', 'W', ' ', 'W', ' '],
        [' ', ' ', ' ', 'W', 'G']
    ], dtype=str)
