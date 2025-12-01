import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from env import GridEnvironment, load_default_grid


# GLOBAL ROOT - starts hidden but can be shown
_hidden_root = None

def _get_root():
    global _hidden_root
    if _hidden_root is None:
        _hidden_root = tk.Tk()
        _hidden_root.withdraw()
    return _hidden_root


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
        colors = {" ": "white", "W": "gray", "S": "green", "G": "red"}
        for r in range(rows):
            for c in range(cols):
                x1, y1 = c * current_cell_size, r * current_cell_size
                x2, y2 = x1 + current_cell_size, y1 + current_cell_size
                canvas.create_rectangle(x1, y1, x2, y2, fill=colors[grid[r, c]], outline="black")

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
    colors = {" ": "white", "W": "gray", "S": "green", "G": "red"}
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
                                       fill=colors[grid[r, c]], outline="black")
        # Create and return agent oval
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
