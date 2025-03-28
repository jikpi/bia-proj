import random

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

from ex4_qlearning.qlearning import Ql


def create_complex_maze(height=20, width=20, obstacle_density=0.25,
                        start_point=(0, 0), goal_point=None):
    maze = np.zeros((height, width), dtype=int)

    if goal_point is None:
        goal_point = (height - 1, width - 1)

    maze[goal_point] = 100

    obstacle_count = int((height * width) * obstacle_density)
    obstacles_placed = 0

    while obstacles_placed < obstacle_count:
        row = random.randint(0, height - 1)
        col = random.randint(0, width - 1)

        if (row, col) != start_point and (row, col) != goal_point and maze[row, col] == 0:
            maze[row, col] = -1
            obstacles_placed += 1

    for i in range(height):
        if i % 4 == 0:
            for j in range(width):
                if maze[i, j] == -1:
                    if random.random() < 0.7:
                        maze[i, j] = 0

    for j in range(width):
        if j % 4 == 0:
            for i in range(height):
                if maze[i, j] == -1:
                    if random.random() < 0.7:
                        maze[i, j] = 0

    return maze


def create_simple_maze(height=5, width=5):
    maze = np.zeros((height, width), dtype=int)

    maze[1, 1] = -1
    maze[1, 3] = -1
    maze[1, 4] = -1
    maze[2, 1] = -1
    maze[3, 3] = -1
    maze[3, 1] = -1

    maze[4, 4] = 100

    return maze


def maze_solve():
    maze_map = create_complex_maze(
        height=10,
        width=10,
        obstacle_density=0.3
    )

    ql = Ql(maze_map)

    print("Exploring the maze...")
    ql.explore(
        iterations=200000,
        learning_rate=0.8,
        initial_state=0,
        epsilon_start=0.9,
        epsilon_end=0.3,
        random_teleport_probability=0.05
    )

    print("Finding path to reward...")
    start_position = (0, 0)
    positions_history = ql.find(start_position, max_iterations=70)

    final_pos = positions_history[-1]
    if maze_map[final_pos.row, final_pos.col] == 100:
        print(f"\nReward found at position ({final_pos.row}, {final_pos.col}) "
              f"after {final_pos.iteration} steps!")
    else:
        print("\nFailed to find reward.")

    print("Creating animation...")
    anim = animate_path(maze_map, positions_history)
    anim.save(f'Outputs/maze_explorer.gif', writer='pillow', fps=2)


def visualize_maze(maze_map, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    displayed_map = maze_map.copy()

    cmap = ListedColormap(['black', 'white', 'green'])

    bounds = [-1.5, -0.5, 0.5, 99.5]
    norm = BoundaryNorm(bounds, cmap.N)

    displayed_map = np.flipud(displayed_map)

    ax.imshow(displayed_map, cmap=cmap, norm=norm)

    for x in range(maze_map.shape[1] + 1):
        ax.axvline(x - 0.5, color='gray', linestyle='-', linewidth=0.5)
    for y in range(maze_map.shape[0] + 1):
        ax.axhline(y - 0.5, color='gray', linestyle='-', linewidth=0.5)

    ax.set_xlim(-0.5, maze_map.shape[1] - 0.5)
    ax.set_ylim(-0.5, maze_map.shape[0] - 0.5)

    ax.set_xticks([])
    ax.set_yticks([])

    for i in range(maze_map.shape[0]):
        for j in range(maze_map.shape[1]):
            display_i = maze_map.shape[0] - 1 - i

            if maze_map[i, j] == -1:
                value = "X"
                color = 'white'
            elif maze_map[i, j] == 100:
                value = "R"
                color = 'black'
            else:
                value = ""
                color = 'black'

            if value:
                ax.text(j, display_i, value, ha='center', va='center', color=color, fontweight='bold')

    return ax


def animate_path(maze_map, positions_history):
    explorer = '🐭'
    fig, ax = plt.subplots(figsize=(8, 8))

    visualize_maze(maze_map, ax)

    agent_marker = ax.text(0, 0, explorer, ha='center', va='center', fontsize=15)

    iteration_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, color='black',
                             fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    def init():
        agent_marker.set_position((0, 0))
        agent_marker.set_text("")
        iteration_text.set_text('')
        return agent_marker, iteration_text

    def animate(i):
        if i < len(positions_history):
            position = positions_history[i]

            displayed_row = maze_map.shape[0] - 1 - position.row

            agent_marker.set_position((position.col, displayed_row))
            agent_marker.set_text(explorer)

            iteration_text.set_text(f'Iteration: {position.iteration}')

        return agent_marker, iteration_text

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(positions_history), interval=500, blit=True
    )

    plt.title("Maze Explorer")

    return anim
