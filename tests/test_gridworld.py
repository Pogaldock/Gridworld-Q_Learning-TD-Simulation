import numpy as np
from env import GridEnvironment, load_default_grid
from agent import RLAgent


def test_transition_block_and_oob():
    grid = np.array([
        ['S', 'G'],
    ], dtype=str)
    env = GridEnvironment(grid)
    # moving left from (0,0) should stay (OOB)
    assert env.transition((0, 0), 'left') == (0, 0)
    # moving right from start should reach goal
    assert env.transition((0, 0), 'right') == (0, 1)


def test_bfs_shortest_path_simple():
    grid = np.array([
        ['S', 'G'],
    ], dtype=str)
    env = GridEnvironment(grid)
    path = env.bfs_shortest_path()
    assert path == [(0, 0), (0, 1)]


def test_rlagent_evaluate_success():
    # Simple 2-cell grid where action 'right' should lead to the goal
    grid = np.array([
        ['S', 'G'],
    ], dtype=str)
    env = GridEnvironment(grid)
    agent = RLAgent(env, max_steps=10)
    states = agent.init_q()
    # set q so that 'right' is best everywhere
    for s in states:
        for a in agent.actions:
            agent.q[(s, a)] = 0.0
        agent.q[(s, 'right')] = 1.0

    stats = agent.evaluate(trials=5)
    assert stats['success_count'] == 5
    assert stats['min'] == 1
