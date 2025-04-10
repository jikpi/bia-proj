import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from ex7_ifs.ifs import Ifs


def visualize_fractal_3d(fractal: Ifs, title: str, num_frames: int = 120, dpi: int = 150):
    points = fractal.point_history[:, 1:]

    num_points = points.shape[1]
    point_size = max(0.1, min(1.0, 10000 / num_points))

    min_vals = np.min(points, axis=1, keepdims=True)
    max_vals = np.max(points, axis=1, keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    normalized_points = (points - min_vals) / range_vals

    fig = plt.figure(figsize=(10, 8), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')

    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    rotation_zone_color = '#0A0A0A'
    ax.xaxis.pane.set_facecolor(rotation_zone_color)
    ax.yaxis.pane.set_facecolor(rotation_zone_color)
    ax.zaxis.pane.set_facecolor(rotation_zone_color)

    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')

    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    ax.set_title(title, color='white', fontsize=16)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.xaxis.line.set_color('white')
    ax.yaxis.line.set_color('white')
    ax.zaxis.line.set_color('none')

    scatter = ax.scatter(
        normalized_points[0],
        normalized_points[1],
        normalized_points[2],
        s=point_size,
        c=np.arange(normalized_points.shape[1]),
        cmap='cool',
        alpha=0.7
    )

    margin = 0.1
    ax.set_xlim(0 - margin, 1 + margin)
    ax.set_ylim(0 - margin, 1 + margin)
    ax.set_zlim(0 - margin, 1 + margin)

    ax.set_box_aspect([1, 1, 1])

    def update(frame):
        ax.view_init(elev=20, azim=frame * (360 / num_frames))
        return scatter,

    animation = FuncAnimation(
        fig, update, frames=num_frames,
        interval=50, blit=True
    )

    output_path = os.path.join('Outputs', f'IFS_{title}_3D.gif')
    animation.save(output_path, dpi=dpi, writer='pillow')

    plt.close()


def visualize_fractal_2d(fractal: Ifs, title: str, num_frames: int = 120, dpi: int = 150):
    points = fractal.point_history[:, 1:]

    num_points = points.shape[1]
    point_size = max(0.1, min(1.0, 10000 / num_points))

    x_points = points[0]
    y_points = points[1]

    min_x, max_x = np.min(x_points), np.max(x_points)
    min_y, max_y = np.min(y_points), np.max(y_points)

    range_x = max_x - min_x if max_x > min_x else 1.0
    range_y = max_y - min_y if max_y > min_y else 1.0

    normalized_x = (x_points - min_x) / range_x
    normalized_y = (y_points - min_y) / range_y

    fig = plt.figure(figsize=(10, 8), facecolor='black')
    ax = fig.add_subplot(111)

    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_title(f"{title}", color='white', fontsize=16)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    scatter = ax.scatter(
        [], [],
        s=point_size,
        c=[],
        cmap='cool',
        alpha=0.7
    )

    margin = 0.1
    ax.set_xlim(0 - margin, 1 + margin)
    ax.set_ylim(0 - margin, 1 + margin)

    ax.set_aspect('equal')

    def update(frame):
        points_to_show = int((frame + 1) / num_frames * len(normalized_x))
        points_to_show = max(1, points_to_show)
        scatter.set_offsets(np.column_stack((
            normalized_x[:points_to_show],
            normalized_y[:points_to_show]
        )))

        scatter.set_array(np.arange(points_to_show))

        return scatter,

    animation = FuncAnimation(
        fig, update, frames=num_frames,
        interval=50, blit=True
    )

    output_path = os.path.join('Outputs', f'IFS_{title}_2D.gif')
    animation.save(output_path, dpi=dpi, writer='pillow')

    plt.close()


def visualize_fractal(fractal: Ifs, title: str, num_frames: int = 120, dpi: int = 150):
    visualize_fractal_3d(fractal, title, num_frames, dpi)
    visualize_fractal_2d(fractal, title, num_frames, dpi)


def ifs_solve():
    first_model = np.array([
        [0.00, 0.00, 0.01, 0.00, 0.26, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00],
        [0.20, -0.26, -0.01, 0.23, 0.22, -0.07, 0.07, 0.00, 0.24, 0.00, 0.80, 0.00],
        [-0.25, 0.28, 0.01, 0.26, 0.24, -0.07, 0.07, 0.00, 0.24, 0.00, 0.22, 0.00],
        [0.85, 0.04, -0.01, -0.04, 0.85, 0.09, 0.00, 0.08, 0.84, 0.00, 0.80, 0.00]
    ])

    second_model = np.array([
        [0.05, 0.00, 0.00, 0.00, 0.60, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00],
        [0.45, -0.22, 0.22, 0.22, 0.45, 0.22, -0.22, 0.22, -0.45, 0.00, 1.00, 0.00],
        [-0.45, 0.22, -0.22, 0.22, 0.45, 0.22, 0.22, -0.22, 0.45, 0.00, 1.25, 0.00],
        [0.49, -0.08, 0.08, 0.08, 0.49, 0.08, 0.08, -0.08, 0.49, 0.00, 2.00, 0.00]
    ])

    dragon_model = np.array([
        [1 / np.sqrt(2), -1 / np.sqrt(2), 0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0, 0, 1, 0, 0, 0],
        [-1 / np.sqrt(2), -1 / np.sqrt(2), 0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0, 0, 0, 1, 1, 0, 0]
    ])

    sierpinski_triangle = np.array([
        [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.25, 0.433, 0.0],
        [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.25, 0.144, 0.35]
    ])

    menger_sponge = np.array([
        [1 / 3, 0, 0, 0, 1 / 3, 0, 0, 0, 1 / 3, dx, dy, dz]
        for dx in [0, 1 / 3, 2 / 3]
        for dy in [0, 1 / 3, 2 / 3]
        for dz in [0, 1 / 3, 2 / 3]
        if (dx, dy, dz) not in [(1 / 3, 1 / 3, 0), (1 / 3, 1 / 3, 1 / 3), (1 / 3, 1 / 3, 2 / 3),
                                (1 / 3, 0, 1 / 3), (1 / 3, 2 / 3, 1 / 3), (0, 1 / 3, 1 / 3),
                                (2 / 3, 1 / 3, 1 / 3)]
    ])

    models = {
        "First Model": (first_model, 10000),
        "Second Model": (second_model, 10000),
        "Dragon Curve": (dragon_model, 10000),  # => cool 2D animace
        "Sierpinski triangle": (sierpinski_triangle, 10000),
        "Menger Sponge": (menger_sponge, 10000),
    }

    print("The animations might take a while.")
    for name, (model, steps) in models.items():
        fractal = Ifs(model, seed=96)
        fractal.step(steps)
        print(f"For {name} generated {fractal.point_history.shape[1]} points")
        print(f"Animating {name}...")
        visualize_fractal(fractal, name)
