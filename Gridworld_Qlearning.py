"""Top-level runner for Gridworld Q-learning simulation.

This file wires together `env`, `agent` and `ui` modules.
"""

from env import load_default_grid, GridEnvironment
from agent import RLAgent
from ui import (
    parameter_panel, build_grid_gui, draw_world,
    animate_episode_comparison, path_policy_only, show_evaluation_popup,
    _get_root
)
import tkinter as tk
from tkinter import ttk


def run_simulation(world, root):
    """Run the simulation with the given world grid."""
    print("Opening parameter panel...")
    params = parameter_panel()
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

    # Create a simple loading/progress window while training runs
    loading = tk.Toplevel(root)
    loading.title("Training...")
    loading.geometry("360x100")
    loading.transient(root)  # keep above
    loading.lift()
    loading.focus_force()
    lbl = tk.Label(loading, text=f"Training... 0/{episodes}", font=("Arial", 11))
    lbl.pack(pady=(12, 6))
    progress = ttk.Progressbar(loading, orient="horizontal", length=300, mode="determinate", maximum=episodes)
    progress.pack(padx=10, pady=(0, 12))

    def progress_cb(done, total):
        # Update progress bar and label
        progress['value'] = done
        lbl.config(text=f"Training... {done}/{total}")
        # Ensure UI updates while training
        try:
            loading.update_idletasks()
        except Exception:
            pass

    q, actions, states, recorded_paths = agent.train(episodes=episodes, progress_callback=progress_cb)
    print("Training complete!")

    # destroy loading window once training is finished
    try:
        loading.destroy()
    except Exception:
        pass

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

    print("Opening world visualization (matplotlib)...")
    draw_world(world, policy_full, path_policy, final_policy_path, q_table=q, actions=actions)

    print("Opening episode comparison animation...")
    animate_episode_comparison(world, recorded_paths, bfs_path, final_policy_path, max_steps=int(params.get("max_steps", 0)), rerun_callback=lambda: run_simulation(world, root))


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

