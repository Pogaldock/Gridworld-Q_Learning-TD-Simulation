import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from env import GridEnvironment, load_default_grid

# Import from new refactored modules
import config
from utils import load_mouse_image, get_direction_angle, calculate_cell_size
from maze_generators import MAZE_GENERATORS


# GLOBAL ROOT - starts hidden but can be shown
_hidden_root = None

def _get_root():
    global _hidden_root
    if _hidden_root is None:
        _hidden_root = tk.Tk()
        _hidden_root.withdraw()
    return _hidden_root

# Compatibility wrappers for old function names
def _load_mouse_image(cell_size, angle=0):
    """Wrapper for backward compatibility."""
    return load_mouse_image(cell_size, angle)

def _get_direction_angle(from_pos, to_pos):
    """Wrapper for backward compatibility."""
    return get_direction_angle(from_pos, to_pos)


def make_fullscreen(fig):
    try:
        mgr = plt.get_current_fig_manager()
        mgr.window.showMaximized()
    except Exception:
        try:
            mgr.full_screen_toggle()
        except Exception:
            pass


def build_grid_gui():
    """Interactive grid builder with full-screen modal window.

    Uses a `Toplevel` attached to the hidden root so we don't create multiple
    `Tk` instances. The window is modal and returns the constructed grid when
    the user clicks Done.
    """
    print("   Grid builder window should now be visible!")
    print("   -> Draw your grid, then click the 'Done' button to continue.")
    
    root = _get_root()
    win = tk.Toplevel(root)
    win.title("Grid Builder - Draw your grid and click DONE when ready")

    rows, cols = 5, 5
    cell_size = 40
    current_tool = tk.StringVar(value="wall")

    # Left control panel
    control = tk.Frame(win)
    control.grid(row=0, column=0, sticky="ns", padx=10, pady=10)

    # Instructions at top
    tk.Label(control, text="Grid Builder", font=("Arial", 14, "bold"), fg="blue").pack(anchor="w", pady=(0, 5))
    tk.Label(control, text="1. Select a tool below", font=("Arial", 9)).pack(anchor="w")
    tk.Label(control, text="2. Click/drag on canvas", font=("Arial", 9)).pack(anchor="w")
    tk.Label(control, text="3. Add Start (S) and Goal (G)", font=("Arial", 9)).pack(anchor="w")
    tk.Label(control, text="4. Click DONE when ready", font=("Arial", 9, "bold")).pack(anchor="w", pady=(0, 10))

    # Canvas where the grid is drawn
    canvas = tk.Canvas(win, bg="white")
    canvas.grid(row=0, column=1, rowspan=15, sticky="nsew")

    # allow resizing
    win.grid_columnconfigure(1, weight=1)
    win.grid_rowconfigure(0, weight=1)

    grid = np.full((rows, cols), ' ', dtype=str)

    def get_cell_size():
        """Calculate cell size to fit canvas while maintaining aspect ratio."""
        canvas.update_idletasks()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            return cell_size
        size_by_width = canvas_width // cols
        size_by_height = canvas_height // rows
        return max(10, min(size_by_width, size_by_height))

    def redraw():
        canvas.delete("all")
        current_cell_size = get_cell_size()
        for r in range(rows):
            for c in range(cols):
                x1, y1 = c * current_cell_size, r * current_cell_size
                x2, y2 = x1 + current_cell_size, y1 + current_cell_size
                canvas.create_rectangle(x1, y1, x2, y2, fill=config.GRID_COLORS[grid[r, c]], outline="black")

    def on_click(event):
        nonlocal grid
        current_cell_size = get_cell_size()
        c = event.x // current_cell_size
        r = event.y // current_cell_size
        if not (0 <= r < rows and 0 <= c < cols):
            return
        tool = current_tool.get()
        if tool == "empty":
            grid[r, c] = ' '
        elif tool == "wall":
            grid[r, c] = 'W'
        elif tool == "start":
            grid[grid == 'S'] = ' '
            grid[r, c] = 'S'
        elif tool == "goal":
            grid[grid == 'G'] = ' '
            grid[r, c] = 'G'
        redraw()

    def on_drag(event):
        """Handle mouse drag painting while left button is held."""
        nonlocal grid
        current_cell_size = get_cell_size()
        c = event.x // current_cell_size
        r = event.y // current_cell_size
        if not (0 <= r < rows and 0 <= c < cols):
            return
        tool = current_tool.get()
        # Avoid repeated updates if cell already has desired value
        prev = grid[r, c]
        if tool == "empty":
            new = ' '
        elif tool == "wall":
            new = 'W'
        elif tool == "start":
            # on drag, don't repeatedly clear previous S; place S only if different
            new = 'S'
        elif tool == "goal":
            new = 'G'
        else:
            new = prev

        if new == prev:
            return

        if tool == "start":
            grid[grid == 'S'] = ' '
            grid[r, c] = 'S'
        elif tool == "goal":
            grid[grid == 'G'] = ' '
            grid[r, c] = 'G'
        else:
            grid[r, c] = new

        redraw()

    canvas.bind("<Button-1>", on_click)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<Configure>", lambda e: redraw())  # Redraw on window resize

    # ---------------- control widgets ----------------
    tk.Label(control, text="Tools", font=("Arial", 12, "bold")).pack(anchor="w")

    def make_tool_button(name, val):
        return tk.Button(control, text=name, width=12, command=lambda: current_tool.set(val))

    make_tool_button("Empty", "empty").pack(pady=2)
    make_tool_button("Wall", "wall").pack(pady=2)
    make_tool_button("Start", "start").pack(pady=2)
    make_tool_button("Goal", "goal").pack(pady=2)

    tk.Label(control, text="", height=1).pack()
    tk.Label(control, text="Current tool:", font=("Arial", 10)).pack(anchor="w")
    lbl = tk.Label(control, textvariable=current_tool, font=("Arial", 10, "bold"))
    lbl.pack(anchor="w", pady=(0, 6))

    # Grid size controls
    tk.Label(control, text="Grid Size", font=("Arial", 12, "bold")).pack(anchor="w", pady=(8, 0))
    size_frame = tk.Frame(control)
    size_frame.pack(anchor="w", pady=4)
    tk.Label(size_frame, text="Rows:").grid(row=0, column=0)
    entry_r = tk.Entry(size_frame, width=4)
    entry_r.insert(0, str(rows))
    entry_r.grid(row=0, column=1, padx=(4, 8))
    tk.Label(size_frame, text="Cols:").grid(row=0, column=2)
    entry_c = tk.Entry(size_frame, width=4)
    entry_c.insert(0, str(cols))
    entry_c.grid(row=0, column=3, padx=(4, 0))

    def resize():
        nonlocal rows, cols, grid, canvas
        try:
            nr = int(entry_r.get())
            nc = int(entry_c.get())
            if 2 <= nr <= 100 and 2 <= nc <= 100:
                rows, cols = nr, nc
                # expand or shrink grid while preserving existing content in top-left
                new = np.full((rows, cols), ' ', dtype=str)
                rmin = min(new.shape[0], grid.shape[0])
                cmin = min(new.shape[1], grid.shape[1])
                new[:rmin, :cmin] = grid[:rmin, :cmin]
                grid = new
                redraw()
        except Exception:
            pass

    tk.Button(control, text="Resize", command=resize).pack(pady=(4, 8))

    # Maze generation functions using new refactored module
    def apply_maze(generator_name):
        """Apply a maze generator from the refactored module."""
        nonlocal grid
        generator_func = MAZE_GENERATORS[generator_name]
        grid[:] = generator_func(rows, cols)
        redraw()
    
    # Maze generation menu
    tk.Label(control, text="Maze Generation", font=("Arial", 12, "bold")).pack(anchor="w", pady=(12, 4))
    
    # Use generators from the refactored module
    maze_algos = [
        ("Classic Maze", "Recursive Backtracking"),
        ("Binary Tree", "Binary Tree"),
        ("Dense Maze", "Prim's Algorithm"),
        ("Open Rooms", "Open Rooms"),
        ("Spiral", "Spiral")
    ]
    
    for display_name, generator_name in maze_algos:
        tk.Button(control, text=display_name, command=lambda gen=generator_name: apply_maze(gen), 
                 width=15, font=("Arial", 9)).pack(pady=2)
    tk.Button(control, text="Clear", command=lambda: clear_grid()).pack(pady=2)
    
    def validate_and_close():
        # Check if grid has both S and G
        has_start = 'S' in grid
        has_goal = 'G' in grid
        if not has_start or not has_goal:
            from tkinter import messagebox
            messagebox.showwarning("Incomplete Grid", 
                                   "Please add both a Start (S) cell and a Goal (G) cell before clicking Done.")
        else:
            win.destroy()
    
    tk.Button(control, text="Done", command=validate_and_close, bg="lightgreen", font=("Arial", 10, "bold")).pack(pady=12)

    def clear_grid():
        nonlocal grid
        grid[:] = ' '
        redraw()

    redraw()
    # Make modal: keep focus on this window until closed
    win.transient(root)
    win.grab_set()
    
    # Force window to appear and update
    win.update_idletasks()
    win.update()
    win.deiconify()  # Ensure window is visible
    win.state("zoomed")  # Maximize window after it's fully initialized
    win.lift()
    win.focus_force()
    
    print("   Waiting for you to click 'Done' in the grid builder...")
    win.wait_window()
    print("   Grid builder closed. Grid received.")
    return grid


def parameter_panel(defaults=None):
    if defaults is None:
        defaults = {
            "episodes": 5000,
            "max_steps": 100,
            "epsilon": 0.1,
            "epsilon_end": 0.01,
            "seed": 0,
            "eval_trials": 50,
            "alpha": 0.01,
            "gamma": 0.95
        }

    root = _get_root()
    panel = tk.Toplevel(root)
    panel.title("RL Parameters")
    panel.state("zoomed")  # Maximize window
    
    # Force window to appear
    panel.update_idletasks()
    panel.update()
    panel.deiconify()
    panel.lift()
    panel.focus_force()
    panel.grab_set()
    
    # Create a frame to center the content
    main_frame = tk.Frame(panel)
    main_frame.pack(expand=True)

    labels = {
        "episodes": "Number of episodes",
        "max_steps": "Max steps per episode",
        "epsilon": "Exploration rate (epsilon)",
        "epsilon_end": "Final epsilon (epsilon_end)",
        "seed": "Random seed (optional)",
        "eval_trials": "Evaluation trials (greedy)",
        "alpha": "Learning rate (alpha)",
        "gamma": "Discount factor (gamma)"
    }

    entries = {}
    row = 0
    for key, label in labels.items():
        tk.Label(main_frame, text=label, font=("Arial", 16)).grid(row=row, column=0, pady=10, sticky="w", padx=20)
        ent = tk.Entry(main_frame, width=15, font=("Arial", 16))
        ent.insert(0, str(defaults[key]))
        ent.grid(row=row, column=1, pady=10, padx=20)
        entries[key] = ent
        row += 1

    params = {}

    def accept():
        for key, entry in entries.items():
            t = entry.get().strip()
            try:
                val = int(t)
            except:
                try:
                    val = float(t)
                except:
                    val = defaults[key]
            params[key] = val
        panel.destroy()

    tk.Button(main_frame, text="Run", font=("Arial", 16, "bold"), command=accept, bg="lightgreen", width=20, height=2).grid(row=row, column=0, columnspan=2, pady=30)
    panel.wait_window()
    return params


def path_policy_only(policy_full: np.ndarray, path: List[Tuple[int, int]]) -> np.ndarray:
    P = np.full(policy_full.shape, ' ', dtype=str)
    for r, c in path:
        val = policy_full[r, c]
        if val in ['U', 'D', 'L', 'R', 'S', 'G']:
            P[r, c] = val
    return P


def draw_world(grid: np.ndarray, policy_full: np.ndarray, path_policy: np.ndarray, path: List[Tuple[int, int]], q_table: dict, actions: list):
    rows, cols = grid.shape
    colors = {' ': '#EEE', 'W': 'black', 'S': 'green', 'G': 'red'}
    value_map = np.zeros((rows, cols), dtype=float)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 'W':
                value_map[r, c] = np.nan
            else:
                value_map[r, c] = max(q_table[((r, c), a)] for a in actions)

    reward_map = np.zeros((rows, cols), dtype=float)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 'W':
                reward_map[r, c] = np.nan
            elif grid[r, c] == 'G':
                reward_map[r, c] = 1.0
            else:
                reward_map[r, c] = 0.0

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(30, 8))
    masked = np.ma.masked_invalid(value_map)
    cmap = plt.cm.inferno.copy()
    cmap.set_bad("gray")
    im = ax1.imshow(masked, cmap=cmap)
    fig.colorbar(im, ax=ax1)
    ax1.set_title("Value Heatmap")
    ax1.set_xticks([]); ax1.set_yticks([])

    masked2 = np.ma.masked_invalid(reward_map)
    cmap2 = plt.cm.plasma.copy()
    cmap2.set_bad("gray")
    im2 = ax2.imshow(masked2, cmap=cmap2)
    fig.colorbar(im2, ax=ax2)
    ax2.set_title("Reward Heatmap")
    ax2.set_xticks([]); ax2.set_yticks([])

    arrows = {'U': (0, -0.3), 'D': (0, 0.3), 'L': (-0.3, 0), 'R': (0.3, 0)}

    def draw_bg(ax):
        for r in range(rows):
            for c in range(cols):
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color=colors.get(grid[r, c], '#EEE'), ec="gray"))

    ax3.set_title("Full Policy")
    draw_bg(ax3)
    for r in range(rows):
        for c in range(cols):
            p = policy_full[r, c]
            if p in arrows:
                dx, dy = arrows[p]
                ax3.arrow(c + 0.5, r + 0.5, dx, dy, head_width=0.2, head_length=0.2, fc="blue", ec="blue")
    ax3.set_xlim(0, cols); ax3.set_ylim(rows, 0)
    ax3.set_xticks([]); ax3.set_yticks([])

    ax4.set_title("Optimal Path Only")
    draw_bg(ax4)
    for r in range(rows):
        for c in range(cols):
            p = path_policy[r, c]
            if p in arrows:
                dx, dy = arrows[p]
                ax4.arrow(c + 0.5, r + 0.5, dx, dy, head_width=0.2, head_length=0.2, fc="blue", ec="blue")
    if len(path) >= 2:
        xs = [c + 0.5 for (_, c) in path]
        ys = [r + 0.5 for (r, _) in path]
        ax4.plot(xs, ys, color="cyan", linewidth=4)
    ax4.set_xlim(0, cols); ax4.set_ylim(rows, 0)
    ax4.set_xticks([]); ax4.set_yticks([])

    plt.tight_layout()
    make_fullscreen(fig)
    plt.show()


def animate_episode_comparison(grid: np.ndarray, recorded_paths: dict, bfs_path: List[Tuple[int, int]], final_policy_path: List[Tuple[int, int]], speed=80, max_steps=None, rerun_callback=None):
    rows, cols = grid.shape
    episodes = sorted(recorded_paths.keys())
    root = _get_root()
    win = tk.Toplevel(root)
    win.title("Episode Comparison")
    win.state("zoomed")  # Maximize window
    win.lift()
    win.focus_force()

    def _quit_all():
        win.destroy()
        import sys
        sys.exit()

    win.protocol("WM_DELETE_WINDOW", _quit_all)
    
    # Main container with button at top
    container = tk.Frame(win)
    container.pack(fill="both", expand=True)
    
    # Button frame at the top
    if rerun_callback:
        button_frame = tk.Frame(container)
        button_frame.pack(side="top", fill="x", padx=10, pady=10)
        
        def on_rerun():
            win.destroy()
            rerun_callback()
        
        tk.Button(button_frame, text="Rerun with New Parameters (Keep Same Maze)", 
                 font=("Arial", 14, "bold"), bg="lightblue", command=on_rerun,
                 height=2).pack(side="top", pady=5)
    
    frame = tk.Frame(container)
    frame.pack(fill="both", expand=True)
    canvases = {}
    agents = {}

    bfs_found = (len(bfs_path) > 0 and grid[bfs_path[-1]] == "G")
    bfs_length = len(bfs_path) - 1 if bfs_found else None
    final_found = (len(final_policy_path) > 0 and grid[final_policy_path[-1]] == "G")
    final_policy_steps = len(final_policy_path) - 1 if final_found else None

    def get_canvas_cell_size(canvas, rows, cols):
        """Calculate cell size for a canvas based on available space."""
        canvas.update_idletasks()
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w <= 1 or h <= 1:
            return 40
        return max(10, min(w // cols, h // rows))

    def draw_episode_canvas(canvas, episode_num, rows, cols, grid, colors):
        """Draw grid on canvas with dynamic cell sizing."""
        canvas.delete("all")
        cell = get_canvas_cell_size(canvas, rows, cols)
        for r in range(rows):
            for c in range(cols):
                canvas.create_rectangle(c * cell, r * cell, c * cell + cell, r * cell + cell, 
                                       fill=config.GRID_COLORS[grid[r, c]], outline="black")
        # Create and return agent image or oval fallback
        mouse_photo = _load_mouse_image(cell)
        if mouse_photo:
            agent = canvas.create_image(cell // 2, cell // 2, image=mouse_photo)
            canvas.mouse_photo = mouse_photo  # Keep reference
        else:
            agent = canvas.create_oval(5, 5, cell - 5, cell - 5, fill="cyan", outline="blue", width=2)
        return agent

    for i, ep in enumerate(episodes):
        ep_frame = tk.Frame(frame)
        ep_frame.grid(row=0, column=i, sticky="nsew", padx=10, pady=10)
        frame.grid_columnconfigure(i, weight=1)
        frame.grid_rowconfigure(0, weight=1)
        ep_path = recorded_paths[ep]
        ep_steps = len(ep_path) - 1
        goal_reached = (grid[ep_path[-1]] == "G")
        goal_indices = [i for i, pos in enumerate(ep_path) if grid[pos] == 'G']
        best_ep_length = goal_indices[0] if goal_indices else None
        optimal_exists = bfs_found
        optimal_len_display = bfs_length if optimal_exists else 'N/A'
        steps_display = f"{ep_steps}/{max_steps}" if max_steps is not None else f"{ep_steps}"
        if best_ep_length is not None and optimal_exists:
            best_display = f"{best_ep_length}/{bfs_length}"
        elif best_ep_length is not None:
            best_display = f"{best_ep_length}/N/A"
        else:
            best_display = f"N/A/{optimal_len_display}"
        episode_followed_optimal = (best_ep_length is not None and optimal_exists and (best_ep_length == bfs_length))
        info_text = (
            f"Episode {ep}\n"
            f"Episode steps: {steps_display}\n"
            f"Reached goal: {'yes' if goal_reached else 'no'}\n"
            f"Optimal path exists: {'yes' if optimal_exists else 'no'}\n"
            f"Episode followed optimal path: {'yes' if episode_followed_optimal else 'no'}\n"
            f"Best path in episode (n_steps): {best_display}\n"
            f"Optimal (BFS) path length: {optimal_len_display}\n"
            f"Steps to goal in final policy: {final_policy_steps if final_policy_steps is not None else 'N/A'}"
        )
        tk.Label(ep_frame, text=info_text, font=("Arial", 12), justify="left", wraplength=250, anchor="w").pack(pady=5, padx=5, fill="x")
        canvas = tk.Canvas(ep_frame, bg="white", highlightthickness=0)
        canvas.pack(fill="both", expand=True, padx=5, pady=5)
        canvases[ep] = canvas
        # Initial draw
        agent = draw_episode_canvas(canvas, ep, rows, cols, grid, colors)
        agents[ep] = agent
        # Redraw on resize
        def on_resize(event, c=canvas, ep_num=ep):
            agents[ep_num] = draw_episode_canvas(c, ep_num, rows, cols, grid, colors)
        canvas.bind("<Configure>", on_resize)

    def animate_ep(ep, step):
        path = recorded_paths[ep]
        if step >= len(path):
            step = 0
        r, c = path[step]
        canvas = canvases[ep]
        cell = get_canvas_cell_size(canvas, rows, cols)
        
        # Calculate direction angle
        angle = 0
        if step > 0:
            prev_pos = path[step - 1]
            curr_pos = path[step]
            angle = _get_direction_angle(prev_pos, curr_pos)
        
        # Update image rotation and position
        mouse_photo = _load_mouse_image(cell, angle)
        if mouse_photo and hasattr(canvas, 'mouse_photo'):
            canvas.itemconfig(agents[ep], image=mouse_photo)
            canvas.mouse_photo = mouse_photo  # Keep reference
            canvas.coords(agents[ep], c * cell + cell // 2, r * cell + cell // 2)
        else:
            # Try image coords first (center x, y), fallback to oval coords (x1, y1, x2, y2)
            try:
                canvas.coords(agents[ep], c * cell + cell // 2, r * cell + cell // 2)
            except:
                canvas.coords(agents[ep], c * cell + 5, r * cell + 5, c * cell + cell - 5, r * cell + cell - 5)
        win.after(speed, lambda: animate_ep(ep, step + 1))

    for ep in episodes:
        animate_ep(ep, 0)


def show_evaluation_popup(eval_stats: dict, bfs_length: int, final_policy_steps: int, trials: int):
    success_count = eval_stats.get('success_count', 0)
    success_rate = eval_stats.get('success_rate', 0.0)
    min_s = eval_stats.get('min')
    median_s = eval_stats.get('median')
    mean_s = eval_stats.get('mean')

    lines = [f"Trials: {trials}", f"Successes: {success_count}/{trials} ({success_rate*100:.1f}%)"]
    if min_s is not None:
        lines.append(f"Steps to goal (successful runs): min={min_s}, median={median_s}, mean={mean_s:.2f}")
    else:
        lines.append("No successful greedy runs to report step statistics.")
    if bfs_length is not None:
        lines.append(f"BFS shortest path length: {bfs_length}")
    else:
        lines.append("BFS found no path from S to G.")
    if final_policy_steps is not None:
        lines.append(f"Final policy reaches goal in {final_policy_steps} steps")
        if bfs_length is not None:
            lines.append(f"Final vs BFS: {final_policy_steps} steps vs {bfs_length} steps")
    else:
        lines.append("Final policy did not reach the goal when followed greedily.")

    messagebox.showinfo("Post-training evaluation", "\n".join(lines))


def show_final_summary(grid: np.ndarray, eval_stats: dict, bfs_length: int, final_policy_steps: int, 
                       trials: int, current_params: dict, rerun_callback):
    """Display final statistics summary with rerun button."""
    root = _get_root()
    win = tk.Toplevel(root)
    win.title("Training Complete - Final Summary")
    win.state("zoomed")
    win.lift()
    win.focus_force()
    win.grab_set()
    
    # Main container
    container = tk.Frame(win, bg="white")
    container.pack(fill="both", expand=True, padx=20, pady=20)
    
    # Title
    title = tk.Label(container, text="🎉 Training Complete!", font=("Arial", 24, "bold"), bg="white", fg="green")
    title.pack(pady=(20, 30))
    
    # Statistics frame
    stats_frame = tk.Frame(container, bg="white", relief="solid", borderwidth=2)
    stats_frame.pack(fill="both", expand=True, padx=40, pady=20)
    
    # Add padding inside stats frame
    inner_frame = tk.Frame(stats_frame, bg="white")
    inner_frame.pack(fill="both", expand=True, padx=30, pady=30)
    
    # Get statistics
    success_count = eval_stats.get('success_count', 0)
    success_rate = eval_stats.get('success_rate', 0.0)
    min_s = eval_stats.get('min')
    median_s = eval_stats.get('median')
    mean_s = eval_stats.get('mean')
    
    # Display statistics
    tk.Label(inner_frame, text="Final Statistics", font=("Arial", 20, "bold"), bg="white", fg="blue").pack(pady=(0, 20))
    
    # Evaluation results
    eval_frame = tk.LabelFrame(inner_frame, text="Evaluation Results", font=("Arial", 16, "bold"), bg="white", padx=20, pady=15)
    eval_frame.pack(fill="x", pady=10)
    
    tk.Label(eval_frame, text=f"Evaluation Trials: {trials}", font=("Arial", 14), bg="white", anchor="w").pack(fill="x", pady=3)
    tk.Label(eval_frame, text=f"Success Rate: {success_count}/{trials} ({success_rate*100:.1f}%)", 
             font=("Arial", 14), bg="white", anchor="w", fg="green" if success_rate > 0.8 else "orange").pack(fill="x", pady=3)
    
    if min_s is not None:
        tk.Label(eval_frame, text=f"Steps to Goal (successful runs):", font=("Arial", 14, "bold"), bg="white", anchor="w").pack(fill="x", pady=(8, 3))
        tk.Label(eval_frame, text=f"  • Minimum: {min_s}", font=("Arial", 14), bg="white", anchor="w").pack(fill="x", pady=2)
        tk.Label(eval_frame, text=f"  • Median: {median_s}", font=("Arial", 14), bg="white", anchor="w").pack(fill="x", pady=2)
        tk.Label(eval_frame, text=f"  • Mean: {mean_s:.2f}", font=("Arial", 14), bg="white", anchor="w").pack(fill="x", pady=2)
    else:
        tk.Label(eval_frame, text="No successful greedy runs", font=("Arial", 14), bg="white", anchor="w", fg="red").pack(fill="x", pady=3)
    
    # Path comparison
    path_frame = tk.LabelFrame(inner_frame, text="Path Analysis", font=("Arial", 16, "bold"), bg="white", padx=20, pady=15)
    path_frame.pack(fill="x", pady=10)
    
    if bfs_length is not None:
        tk.Label(path_frame, text=f"Optimal Path (BFS): {bfs_length} steps", font=("Arial", 14), bg="white", anchor="w").pack(fill="x", pady=3)
    else:
        tk.Label(path_frame, text="BFS found no path from S to G", font=("Arial", 14), bg="white", anchor="w", fg="red").pack(fill="x", pady=3)
    
    if final_policy_steps is not None:
        tk.Label(path_frame, text=f"Final Policy Path: {final_policy_steps} steps", font=("Arial", 14), bg="white", anchor="w").pack(fill="x", pady=3)
        if bfs_length is not None:
            diff = final_policy_steps - bfs_length
            diff_text = f"Difference: +{diff} steps" if diff > 0 else f"Difference: {diff} steps" if diff < 0 else "Perfect match! ✓"
            color = "green" if diff == 0 else "orange" if diff <= 2 else "red"
            tk.Label(path_frame, text=diff_text, font=("Arial", 14, "bold"), bg="white", anchor="w", fg=color).pack(fill="x", pady=3)
    else:
        tk.Label(path_frame, text="Final policy did not reach goal", font=("Arial", 14), bg="white", anchor="w", fg="red").pack(fill="x", pady=3)
    
    # Training parameters
    params_frame = tk.LabelFrame(inner_frame, text="Training Parameters", font=("Arial", 16, "bold"), bg="white", padx=20, pady=15)
    params_frame.pack(fill="x", pady=10)
    
    param_labels = {
        "episodes": "Episodes",
        "max_steps": "Max Steps per Episode",
        "alpha": "Learning Rate (α)",
        "gamma": "Discount Factor (γ)",
        "epsilon": "Initial Epsilon (ε)",
        "epsilon_end": "Final Epsilon"
    }
    
    for key, label in param_labels.items():
        if key in current_params:
            tk.Label(params_frame, text=f"{label}: {current_params[key]}", font=("Arial", 14), bg="white", anchor="w").pack(fill="x", pady=2)
    
    # Buttons frame - use pack with side to ensure visibility
    button_frame = tk.Frame(container, bg="white")
    button_frame.pack(pady=20, fill="x")
    
    def on_rerun():
        win.destroy()
        rerun_callback()
    
    def on_finish():
        win.destroy()
        import sys
        sys.exit()
    
    # Add window close handler
    win.protocol("WM_DELETE_WINDOW", on_finish)
    
    # Center the buttons
    button_container = tk.Frame(button_frame, bg="white")
    button_container.pack()
    
    tk.Button(button_container, text="Rerun with New Parameters (Same Maze)", 
              font=("Arial", 14, "bold"), bg="lightblue", fg="black",
              command=on_rerun, width=40, height=2, relief="raised", borderwidth=3).pack(pady=8)
    
    tk.Button(button_container, text="Finish", 
              font=("Arial", 14, "bold"), bg="lightgray", fg="black",
              command=on_finish, width=40, height=2, relief="raised", borderwidth=3).pack(pady=8)
    
    win.wait_window()


class TrainingAnimationViewer:
    """Interactive viewer for trained episodes with navigation controls."""
    
    def __init__(self, grid: np.ndarray, all_paths: dict, epsilon_values: dict, max_steps: int, animation_speed: int = 50, bfs_length: int = None):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.all_paths = all_paths  # Dict mapping episode number to path
        self.epsilon_values = epsilon_values  # Dict mapping episode number to epsilon
        self.max_steps = max_steps
        self.animation_speed = animation_speed
        self.bfs_length = bfs_length  # Optimal path length for comparison
        
        self.episode_numbers = sorted(all_paths.keys())
        self.current_episode_idx = 0
        self.current_step = 0
        self.is_playing = True
        self.animation_id = None
        
        # Find first success and first optimal episodes
        self.first_success_ep = None
        self.first_optimal_ep = None
        for ep_num in self.episode_numbers:
            path = self.all_paths[ep_num]
            # Check if reached goal
            if len(path) > 0 and self.grid[path[-1]] == 'G':
                if self.first_success_ep is None:
                    self.first_success_ep = ep_num
                # Check if optimal (reached goal in BFS steps)
                if self.bfs_length is not None and len(path) - 1 == self.bfs_length:
                    if self.first_optimal_ep is None:
                        self.first_optimal_ep = ep_num
                        break  # Found both, can stop searching
        
        root = _get_root()
        self.win = tk.Toplevel(root)
        self.win.title("Training Animation Viewer")
        self.win.state("zoomed")
        self.win.lift()
        
        # Create main container
        container = tk.Frame(self.win)
        container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Info panel at top
        info_frame = tk.Frame(container)
        info_frame.pack(side="top", fill="x", pady=(0, 10))
        
        self.episode_label = tk.Label(info_frame, text="Episode: 0 / 0", font=("Arial", 16, "bold"))
        self.episode_label.pack(side="left", padx=10)
        
        self.step_label = tk.Label(info_frame, text="Step: 0", font=("Arial", 14))
        self.step_label.pack(side="left", padx=10)
        
        self.epsilon_label = tk.Label(info_frame, text="Epsilon: 0.00", font=("Arial", 14))
        self.epsilon_label.pack(side="left", padx=10)
        
        self.status_label = tk.Label(info_frame, text="", font=("Arial", 14))
        self.status_label.pack(side="left", padx=10)
        
        # Control panel
        control_frame = tk.Frame(container)
        control_frame.pack(side="top", fill="x", pady=(0, 10))
        
        # Navigation buttons
        tk.Button(control_frame, text="⏮ First", font=("Arial", 12), command=self.first_episode, width=8).pack(side="left", padx=5)
        tk.Button(control_frame, text="◀ Previous", font=("Arial", 12), command=self.previous_episode, width=10).pack(side="left", padx=5)
        
        self.play_pause_btn = tk.Button(control_frame, text="⏸ Pause", font=("Arial", 12), command=self.toggle_play_pause, width=8)
        self.play_pause_btn.pack(side="left", padx=5)
        
        tk.Button(control_frame, text="Next ▶", font=("Arial", 12), command=self.next_episode, width=10).pack(side="left", padx=5)
        tk.Button(control_frame, text="Last ⏭", font=("Arial", 12), command=self.last_episode, width=8).pack(side="left", padx=5)
        
        # Special jump buttons
        tk.Label(control_frame, text="|", font=("Arial", 12)).pack(side="left", padx=10)
        success_btn = tk.Button(control_frame, text="🎯 First Success", font=("Arial", 12), command=self.goto_first_success, 
                               width=14, bg="lightgreen" if self.first_success_ep else "lightgray",
                               state="normal" if self.first_success_ep else "disabled")
        success_btn.pack(side="left", padx=5)
        
        optimal_btn = tk.Button(control_frame, text="⭐ First Optimal", font=("Arial", 12), command=self.goto_first_optimal, 
                               width=14, bg="gold" if self.first_optimal_ep else "lightgray",
                               state="normal" if self.first_optimal_ep else "disabled")
        optimal_btn.pack(side="left", padx=5)
        
        # Episode jump
        tk.Label(control_frame, text="Go to episode:", font=("Arial", 12)).pack(side="left", padx=(20, 5))
        self.episode_entry = tk.Entry(control_frame, width=8, font=("Arial", 12))
        self.episode_entry.pack(side="left", padx=5)
        tk.Button(control_frame, text="Go", font=("Arial", 12), command=self.jump_to_episode, width=5).pack(side="left", padx=5)
        
        # Speed control
        tk.Label(control_frame, text="Speed:", font=("Arial", 12)).pack(side="left", padx=(20, 5))
        self.speed_var = tk.StringVar(value="Normal")
        speed_menu = tk.OptionMenu(control_frame, self.speed_var, "Slow", "Normal", "Fast", "Very Fast", command=self.change_speed)
        speed_menu.config(font=("Arial", 12))
        speed_menu.pack(side="left", padx=5)
        
        # Canvas for grid
        self.canvas = tk.Canvas(container, bg="white", highlightthickness=2, highlightbackground="black")
        self.canvas.pack(fill="both", expand=True)
        
        self.agent_oval = None
        self.draw_grid()
        
        # Handle resize
        self.canvas.bind("<Configure>", lambda e: self.draw_grid())
        
        # Start animation
        self.update_display()
        self.animate()
        
    def get_cell_size(self):
        """Calculate cell size based on canvas dimensions."""
        self.canvas.update_idletasks()
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1 or h <= 1:
            return 40
        return max(15, min(w // self.cols, h // self.rows))
    
    def draw_grid(self):
        """Draw the grid background."""
        self.canvas.delete("all")
        cell = self.get_cell_size()
        
        for r in range(self.rows):
            for c in range(self.cols):
                x1, y1 = c * cell, r * cell
                x2, y2 = x1 + cell, y1 + cell
                cell_color = config.GRID_COLORS[self.grid[r, c]]
                self.canvas.create_rectangle(x1, y1, x2, y2, 
                                            fill=cell_color, 
                                            outline="black", width=1)
                
                # Draw "E" text on end square
                if self.grid[r, c] == 'G':
                    text_size = max(10, int(cell * 0.6))
                    self.canvas.create_text(c * cell + cell // 2, r * cell + cell // 2,
                                           text="E", font=("Arial", text_size, "bold"),
                                           fill="white")
        
        # Create agent image or oval at start position
        mouse_photo = _load_mouse_image(cell)
        if mouse_photo:
            self.agent_oval = self.canvas.create_image(cell // 2, cell // 2, image=mouse_photo)
            self.canvas.mouse_photo = mouse_photo  # Keep reference
        else:
            self.agent_oval = self.canvas.create_oval(5, 5, cell - 5, cell - 5, 
                                                      fill="cyan", outline="blue", width=3)
        # Update agent position to current step
        self.update_agent_position()
    
    def update_agent_position(self):
        """Update agent position on canvas."""
        episode_num = self.episode_numbers[self.current_episode_idx]
        path = self.all_paths[episode_num]
        
        if self.current_step < len(path):
            r, c = path[self.current_step]
            cell = self.get_cell_size()
            
            # Calculate direction angle
            angle = 0
            if self.current_step > 0:
                prev_pos = path[self.current_step - 1]
                curr_pos = path[self.current_step]
                angle = _get_direction_angle(prev_pos, curr_pos)
            
            # Center position for image or oval
            x_center = c * cell + cell // 2
            y_center = r * cell + cell // 2
            
            # Reload and rotate image based on direction
            mouse_photo = _load_mouse_image(cell, angle)
            if mouse_photo and hasattr(self.canvas, 'mouse_photo'):
                self.canvas.itemconfig(self.agent_oval, image=mouse_photo)
                self.canvas.mouse_photo = mouse_photo  # Keep reference
                self.canvas.coords(self.agent_oval, x_center, y_center)
            else:
                # For image: coords takes center x, y
                # For oval: coords takes x1, y1, x2, y2
                try:
                    self.canvas.coords(self.agent_oval, x_center, y_center)
                except:
                    # Fallback for oval coordinates
                    x1 = c * cell + 5
                    y1 = r * cell + 5
                    x2 = c * cell + cell - 5
                    y2 = r * cell + cell - 5
                    self.canvas.coords(self.agent_oval, x1, y1, x2, y2)
    
    def update_display(self):
        """Update all display labels."""
        episode_num = self.episode_numbers[self.current_episode_idx]
        path = self.all_paths[episode_num]
        epsilon = self.epsilon_values.get(episode_num, 0.0)
        
        total_episodes = len(self.episode_numbers)
        self.episode_label.config(text=f"Episode: {episode_num} / {self.episode_numbers[-1]} (viewing {self.current_episode_idx + 1}/{total_episodes})")
        self.step_label.config(text=f"Step: {self.current_step} / {len(path) - 1}")
        self.epsilon_label.config(text=f"Epsilon: {epsilon:.4f}")
        
        # Status
        if self.current_step >= len(path) - 1:
            reached_goal = self.grid[path[-1]] == 'G'
            status_text = f"Goal reached in {len(path)-1} steps!" if reached_goal else f"Max steps reached ({len(path)-1} steps)"
            self.status_label.config(text=status_text)
        else:
            self.status_label.config(text="")
    
    def animate(self):
        """Animation loop."""
        if self.animation_id:
            self.win.after_cancel(self.animation_id)
        
        if self.is_playing:
            episode_num = self.episode_numbers[self.current_episode_idx]
            path = self.all_paths[episode_num]
            
            if self.current_step < len(path) - 1:
                self.current_step += 1
                self.update_agent_position()
                self.update_display()
            else:
                # Loop current episode
                self.current_step = 0
                self.update_agent_position()
                self.update_display()
        
        self.animation_id = self.win.after(self.animation_speed, self.animate)
    
    def toggle_play_pause(self):
        """Toggle between play and pause."""
        self.is_playing = not self.is_playing
        self.play_pause_btn.config(text="▶ Play" if not self.is_playing else "⏸ Pause")
    
    def first_episode(self):
        """Jump to first episode."""
        self.current_episode_idx = 0
        self.current_step = 0
        self.update_agent_position()
        self.update_display()
    
    def previous_episode(self):
        """Go to previous episode."""
        if self.current_episode_idx > 0:
            self.current_episode_idx -= 1
            self.current_step = 0
            self.update_agent_position()
            self.update_display()
    
    def next_episode(self):
        """Go to next episode."""
        if self.current_episode_idx < len(self.episode_numbers) - 1:
            self.current_episode_idx += 1
            self.current_step = 0
            self.update_agent_position()
            self.update_display()
    
    def last_episode(self):
        """Jump to last episode."""
        self.current_episode_idx = len(self.episode_numbers) - 1
        self.current_step = 0
        self.update_agent_position()
        self.update_display()
    
    def jump_to_episode(self):
        """Jump to specific episode number."""
        try:
            episode_num = int(self.episode_entry.get())
            if episode_num in self.episode_numbers:
                self.current_episode_idx = self.episode_numbers.index(episode_num)
                self.current_step = 0
                self.update_agent_position()
                self.update_display()
            else:
                messagebox.showwarning("Invalid Episode", f"Episode {episode_num} was not recorded.")
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid episode number.")
    
    def change_speed(self, speed_name):
        """Change animation speed."""
        speeds = {
            "Slow": 150,
            "Normal": 50,
            "Fast": 20,
            "Very Fast": 5
        }
        self.animation_speed = speeds.get(speed_name, 50)
    
    def goto_first_success(self):
        """Jump to first episode where agent reached the goal."""
        if self.first_success_ep is not None:
            self.current_episode_idx = self.episode_numbers.index(self.first_success_ep)
            self.current_step = 0
            self.update_agent_position()
            self.update_display()
        else:
            messagebox.showinfo("No Success", "No episode found where the agent reached the goal.")
    
    def goto_first_optimal(self):
        """Jump to first episode where agent took the optimal path."""
        if self.first_optimal_ep is not None:
            self.current_episode_idx = self.episode_numbers.index(self.first_optimal_ep)
            self.current_step = 0
            self.update_agent_position()
            self.update_display()
        else:
            if self.bfs_length is None:
                messagebox.showinfo("No Optimal Path", "No optimal BFS path exists for this maze.")
            else:
                messagebox.showinfo("No Optimal", "No episode found where the agent took the optimal path.")
    
    def close(self):
        """Close the visualization window."""
        if self.animation_id:
            self.win.after_cancel(self.animation_id)
        try:
            self.win.destroy()
        except:
            pass
