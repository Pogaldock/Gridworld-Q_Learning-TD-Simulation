import numpy as np
from env import GridEnvironment
from agent import RLAgent


def test_reward_shaping_and_bump():
    grid = np.array([
        ['S', ' '],
        ['W', 'G']
    ], dtype=str)
    env = GridEnvironment(grid)
    # moving right from S should be a normal step
    assert env.transition((0, 0), 'right') == (0, 1)
    # moving down from (0,0) bumps into wall -> stays
    assert env.transition((0, 0), 'down') == (0, 0)
    # bump reward
    assert env.reward((0, 0), (0, 0)) < 0
    # normal step penalty
    assert env.reward((0, 0), (0, 1)) < 0


def test_bfs_no_path_returns_empty():
    grid = np.array([
        ['S', 'W', 'G']
    ], dtype=str)
    env = GridEnvironment(grid)
    path = env.bfs_shortest_path()
    assert path == []


def test_agent_train_recording_with_seed():
    grid = np.array([
        ['S', 'G']
    ], dtype=str)
    env = GridEnvironment(grid)
    agent = RLAgent(env, max_steps=10, seed=42)
    # request recording at episodes 1 and 3
    q, actions, states, recorded = agent.train(episodes=3, record_eps=[1, 3])
    assert 1 in recorded and 3 in recorded
