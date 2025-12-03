"""Maze generation algorithms for the Gridworld."""
import numpy as np
import random
from typing import Tuple


def place_start_goal(grid: np.ndarray, rows: int, cols: int) -> None:
    """Place start and goal in the maze at empty cells."""
    empty = [(r, c) for r in range(rows) for c in range(cols) if grid[r, c] == ' ']
    if len(empty) >= 2:
        start_pos = random.choice(empty)
        empty.remove(start_pos)
        goal_pos = random.choice(empty)
        grid[start_pos] = 'S'
        grid[goal_pos] = 'G'


def generate_recursive_backtracking(rows: int, cols: int) -> np.ndarray:
    """Generate maze using recursive backtracking - creates long winding paths."""
    grid = np.full((rows, cols), 'W', dtype=str)
    
    def carve_passages(r: int, c: int, visited: set) -> None:
        visited.add((r, c))
        grid[r, c] = ' '
        
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(directions)
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                grid[r + dr // 2, c + dc // 2] = ' '
                carve_passages(nr, nc, visited)
    
    start_r = 1 if rows > 1 else 0
    start_c = 1 if cols > 1 else 0
    carve_passages(start_r, start_c, set())
    place_start_goal(grid, rows, cols)
    return grid


def generate_binary_tree(rows: int, cols: int) -> np.ndarray:
    """Generate maze with diagonal bias - easier but interesting."""
    grid = np.full((rows, cols), 'W', dtype=str)
    
    for r in range(0, rows, 2):
        for c in range(0, cols, 2):
            grid[r, c] = ' '
            
            neighbors = []
            if r > 1:
                neighbors.append((-1, 0))  # North
            if c > 1:
                neighbors.append((0, -1))  # West
            
            if neighbors:
                dr, dc = random.choice(neighbors)
                grid[r + dr, c + dc] = ' '
    
    place_start_goal(grid, rows, cols)
    return grid


def generate_prims_algorithm(rows: int, cols: int) -> np.ndarray:
    """Generate dense maze with many short paths using Prim's algorithm."""
    grid = np.full((rows, cols), 'W', dtype=str)
    
    start_r = random.randint(0, rows - 1)
    start_c = random.randint(0, cols - 1)
    grid[start_r, start_c] = ' '
    
    walls = []
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = start_r + dr, start_c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            walls.append((nr, nc, start_r, start_c))
    
    while walls:
        wall = random.choice(walls)
        walls.remove(wall)
        r, c, pr, pc = wall
        
        if grid[r, c] == 'W':
            opposite_neighbors = 0
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == ' ' and (nr, nc) != (pr, pc):
                    opposite_neighbors += 1
            
            if opposite_neighbors <= 1:
                grid[r, c] = ' '
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 'W':
                        walls.append((nr, nc, r, c))
    
    place_start_goal(grid, rows, cols)
    return grid


def generate_open_rooms(rows: int, cols: int) -> np.ndarray:
    """Generate maze with large open rooms."""
    grid = np.full((rows, cols), ' ', dtype=str)
    
    # Add some random walls
    num_walls = (rows * cols) // 5
    for _ in range(num_walls):
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        grid[r, c] = 'W'
    
    # Add some wall lines
    num_lines = max(2, min(rows, cols) // 3)
    for _ in range(num_lines):
        if random.choice([True, False]):
            # Horizontal line
            r = random.randint(0, rows - 1)
            length = random.randint(2, cols // 2)
            start_c = random.randint(0, cols - length)
            for c in range(start_c, start_c + length):
                grid[r, c] = 'W'
        else:
            # Vertical line
            c = random.randint(0, cols - 1)
            length = random.randint(2, rows // 2)
            start_r = random.randint(0, rows - length)
            for r in range(start_r, start_r + length):
                grid[r, c] = 'W'
    
    place_start_goal(grid, rows, cols)
    return grid


def generate_spiral(rows: int, cols: int) -> np.ndarray:
    """Generate spiral pattern maze."""
    grid = np.full((rows, cols), 'W', dtype=str)
    
    # Create a spiral from outside to center
    top, bottom = 0, rows - 1
    left, right = 0, cols - 1
    
    while top <= bottom and left <= right:
        # Top row
        for c in range(left, right + 1):
            if 0 <= top < rows and 0 <= c < cols:
                grid[top, c] = ' '
        top += 1
        
        # Right column
        for r in range(top, bottom + 1):
            if 0 <= r < rows and 0 <= right < cols:
                grid[r, right] = ' '
        right -= 1
        
        # Bottom row
        if top <= bottom:
            for c in range(right, left - 1, -1):
                if 0 <= bottom < rows and 0 <= c < cols:
                    grid[bottom, c] = ' '
            bottom -= 1
        
        # Left column
        if left <= right:
            for r in range(bottom, top - 1, -1):
                if 0 <= r < rows and 0 <= left < cols:
                    grid[r, left] = ' '
            left += 1
    
    place_start_goal(grid, rows, cols)
    return grid


# Registry of all maze generators
MAZE_GENERATORS = {
    "Recursive Backtracking": generate_recursive_backtracking,
    "Binary Tree": generate_binary_tree,
    "Prim's Algorithm": generate_prims_algorithm,
    "Open Rooms": generate_open_rooms,
    "Spiral": generate_spiral
}
