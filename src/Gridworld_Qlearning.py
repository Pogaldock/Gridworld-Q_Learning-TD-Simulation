"""Top-level runner for Gridworld Q-learning simulation.

This file wires together `env`, `agent` and `ui` modules.
"""

from env import load_default_grid, GridEnvironment
from agent import RLAgent
from ui import (
    parameter_panel, build_grid_gui, draw_world,
    path_policy_only, show_evaluation_popup,
    _get_root, TrainingAnimationViewer, show_final_summary
)
import tkinter as tk
from tkinter import ttk


def run_simulation(world, root, previous_params=None):
    """Run the simulation with the given world grid."""
    print("Opening parameter panel...")
    params = parameter_panel(defaults=previous_params)
    print(f"Parameters set: {params}")

    if not params:
        params = {
            "episodes": 10000,
            "max_steps": 200,
            "epsilon": 0.1,
            "alpha": 0.01,
            "gamma": 0.95
        }

    print("Creating environment and validating...")
    env = GridEnvironment(world)
    try:
        env.validate()
        print(f"Environment validated. Start: {env.start}, Goal: {env.goal}")
    except ValueError as e:
        print(f"Grid validation error: {e}")
        raise

    agent = RLAgent(
        env,
        alpha=float(params.get("alpha", 0.1)),
        gamma=float(params.get("gamma", 0.95)),
        epsilon_start=float(params.get("epsilon", 0.1)),
        epsilon_end=float(params.get("epsilon_end", 0.01)) if params.get("epsilon_end") is not None else 0.01,
        max_steps=int(params.get("max_steps", 100)),
        seed=int(params.get("seed", 0)) if params.get("seed") is not None else None
    )

    episodes = int(params.get("episodes", 1000))
    print(f"Starting training for {episodes} episodes...")
    
    import time
    start_time = time.time()
    last_update = 0
    
    def progress_cb(done, total):
        nonlocal last_update
        current_time = time.time()
        
        # Update every 0.5 seconds or on completion
        if current_time - last_update >= 0.5 or done == total:
            last_update = current_time
            percent = (done / total * 100) if total > 0 else 0
            elapsed = int(current_time - start_time)
            
            # Calculate ETA
            eta_str = ""
            if done > 0 and done < total:
                avg_time = elapsed / done
                remaining = total - done
                eta_seconds = int(avg_time * remaining)
                if eta_seconds < 60:
                    eta_str = f" | ETA: {eta_seconds}s"
                else:
                    eta_minutes = eta_seconds // 60
                    eta_str = f" | ETA: {eta_minutes}m {eta_seconds % 60}s"
            
            # Create progress bar
            bar_length = 40
            filled = int(bar_length * done / total)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            # Print progress (overwrite previous line)
            print(f"\r[{bar}] {percent:.1f}% | Episode {done}/{total} | Elapsed: {elapsed}s{eta_str}", end='', flush=True)
    
    # Train with all episodes recorded
    q, actions, states, recorded_paths, epsilon_values = agent.train(
        episodes=episodes, 
        progress_callback=progress_cb,
        record_all=True
    )
    
    print("\n✓ Training complete!")

    print("Extracting policy and paths...")
    policy_full = agent.extract_policy()
    final_policy_path = agent.trace_policy_path()
    path_policy = path_policy_only(policy_full, final_policy_path)

    bfs_path = env.bfs_shortest_path()

    eval_trials = int(params.get("eval_trials", 50)) if params.get("eval_trials") is not None else 50
    eval_stats = agent.evaluate(trials=eval_trials)

    bfs_length = len(bfs_path) - 1 if len(bfs_path) > 0 else None
    final_policy_steps = len(final_policy_path) - 1 if len(final_policy_path) > 0 and world[final_policy_path[-1]] == 'G' else None

    print("Displaying results...")
    # Show evaluation summary in a popup rather than console
    show_evaluation_popup(eval_stats, bfs_length, final_policy_steps, eval_trials)

    print("Opening interactive training animation viewer...")
    viewer = TrainingAnimationViewer(world, recorded_paths, epsilon_values, max_steps=int(params.get("max_steps", 0)), animation_speed=50, bfs_length=bfs_length)
    viewer.win.wait_window()

    print("Opening world visualization (matplotlib)...")
    draw_world(world, policy_full, path_policy, final_policy_path, q_table=q, actions=actions)

    print("Opening final statistics summary...")
    show_final_summary(world, eval_stats, bfs_length, final_policy_steps, eval_trials, params, lambda: run_simulation(world, root, previous_params=params))


def main():
    print("Starting Gridworld Q-Learning Simulation...")
    root = _get_root()
    print("Root initialized.")
    
    USE_DEFAULT = False
    if USE_DEFAULT:
        print("Using default grid...")
        world = load_default_grid()
    else:
        print("Opening grid builder...")
        world = build_grid_gui()
    print(f"Grid created: {world.shape}")
    
    run_simulation(world, root)
    
    print("Starting main event loop. Close all windows to exit.")
    tk.mainloop()
    print("Program finished.")


if __name__ == "__main__":
    main()

