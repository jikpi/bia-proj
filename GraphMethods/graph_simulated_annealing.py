import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import Type, List

from ex_3_simulated_annealing.simulated_annealing import IterationNeighbourData
from function import OptimizationFunction


def graph_simulated_annealing(optimization_function: Type[OptimizationFunction],
                              iteration_data: List[IterationNeighbourData],
                              initial_point: np.ndarray):
    # 2 ploty
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    ax3d = fig.add_subplot(gs[0], projection='3d')
    ax2d = fig.add_subplot(gs[1])

    # barvy
    fig.patch.set_facecolor('#050505')
    ax3d.set_facecolor('#050505')
    ax2d.set_facecolor('#050505')

    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor('purple')
    ax3d.yaxis.pane.set_edgecolor('purple')
    ax3d.zaxis.pane.set_edgecolor('purple')
    ax3d.grid(True, color='purple', linestyle=':', alpha=0.5)

    # 3D
    ax3d.set_xlabel('X', color='white', fontweight='bold')
    ax3d.set_ylabel('Y', color='white', fontweight='bold')
    ax3d.set_zlabel('Z', color='white', fontweight='bold')
    ax3d.set_title(f'3D View: {optimization_function.__name__}', color='#4cc9f0', fontweight='bold', fontsize=16)

    # 2D
    ax2d.grid(True, color='purple', linestyle=':', alpha=0.5)
    ax2d.set_xlabel('X', color='white', fontweight='bold')
    ax2d.set_ylabel('Y', color='white', fontweight='bold')
    ax2d.set_title(f'2D View: {optimization_function.__name__}', color='#4cc9f0', fontweight='bold', fontsize=16)

    for ax in [ax3d, ax2d]:
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
    ax3d.tick_params(axis='z', colors='white')

    # plot
    x_range, y_range = optimization_function.recommended_range(), optimization_function.recommended_range()

    # osa
    points = 100
    x = np.linspace(x_range[0], x_range[1], points)
    y = np.linspace(y_range[0], y_range[1], points)
    X, Y = np.meshgrid(x, y)

    # evaluace funkce
    Z = np.array(
        [[optimization_function.evaluate(np.array([X[i, j], Y[i, j]])) for j in range(points)] for i in range(points)])

    # plot 3D
    surface = ax3d.plot_surface(X, Y, Z, cmap='winter', antialiased=True, alpha=0.5)

    contour = ax2d.contourf(X, Y, Z, levels=20, cmap='winter', alpha=0.5)

    cbar = fig.colorbar(surface, ax=ax3d, shrink=0.5, aspect=10, pad=0.1)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    current_point_3d = ax3d.scatter([], [], [], color='red', s=100, label='Current')
    neighbors_3d = ax3d.scatter([], [], [], color='yellow', s=50, label='Neighbors')
    best_neighbor_3d = ax3d.scatter([], [], [], color='white', s=100, label='Best')
    accepted_neighbor_3d = ax3d.scatter([], [], [], color='purple', s=100, label='Accepted')

    current_point_2d = ax2d.scatter([], [], color='red', s=100, label='Current')
    neighbors_2d = ax2d.scatter([], [], color='yellow', s=50, label='Neighbors')
    best_neighbor_2d = ax2d.scatter([], [], color='white', s=100, label='Best')
    accepted_neighbor_2d = ax2d.scatter([], [], color='purple', s=100, label='Accepted')

    # legenda
    ax3d.legend(loc='upper right', facecolor='#050505', edgecolor='purple', labelcolor='white')
    ax2d.legend(loc='upper right', facecolor='#050505', edgecolor='purple', labelcolor='white')

    # text pro nejlepsi bod
    best_neighbor_text = fig.text(0.5, 0.02, '', ha='center', va='center', color='white', fontsize=12)

    def update(frame):
        if frame == 0:
            current_point = initial_point
            current_z = optimization_function.evaluate(initial_point)
            current_point_3d._offsets3d = ([current_point[0]], [current_point[1]], [current_z])
            current_point_2d.set_offsets(np.array([current_point[0], current_point[1]]).reshape(1, 2))
            best_neighbor_text.set_text(
                f"Initial point: f({current_point[0]:.2f}, {current_point[1]:.2f}) = {current_z:.2f}")
        else:
            data = iteration_data[frame - 1]

            # sousedi
            x_neighbors = [n[0] for n in data.neighbors]
            y_neighbors = [n[1] for n in data.neighbors]
            z_neighbors = [optimization_function.evaluate(n) for n in data.neighbors]
            neighbors_3d._offsets3d = (x_neighbors, y_neighbors, z_neighbors)
            neighbors_2d.set_offsets(np.column_stack((x_neighbors, y_neighbors)))

            # nejlepsi soused
            best_z = optimization_function.evaluate(data.best_neighbor)
            best_neighbor_3d._offsets3d = ([data.best_neighbor[0]], [data.best_neighbor[1]], [best_z])
            best_neighbor_2d.set_offsets(np.array([data.best_neighbor[0], data.best_neighbor[1]]).reshape(1, 2))

            # accepted soused
            if data.accepted_neighbor is not None:
                accepted_z = optimization_function.evaluate(data.accepted_neighbor)
                accepted_neighbor_3d._offsets3d = (
                    [data.accepted_neighbor[0]], [data.accepted_neighbor[1]], [accepted_z])
                accepted_neighbor_2d.set_offsets(
                    np.array([data.accepted_neighbor[0], data.accepted_neighbor[1]]).reshape(1, 2))
                current_point_3d._offsets3d = ([data.accepted_neighbor[0]], [data.accepted_neighbor[1]], [accepted_z])
                current_point_2d.set_offsets(
                    np.array([data.accepted_neighbor[0], data.accepted_neighbor[1]]).reshape(1, 2))
                best_neighbor_text.set_text(
                    f"Accepted: f({data.accepted_neighbor[0]:.2f}, {data.accepted_neighbor[1]:.2f}) = {accepted_z:.2f}")
            else:
                accepted_neighbor_3d._offsets3d = ([], [], [])
                accepted_neighbor_2d.set_offsets(np.array([]).reshape(0, 2))
                best_neighbor_text.set_text(
                    f"No accepted neighbor in this iteration")

        return current_point_3d, neighbors_3d, best_neighbor_3d, accepted_neighbor_3d, current_point_2d, neighbors_2d, best_neighbor_2d, accepted_neighbor_2d, best_neighbor_text

    plt.tight_layout()

    anim = FuncAnimation(fig, update, frames=len(iteration_data) + 1, interval=1000, blit=False, repeat=True)
    anim.save(f'Outputs/simulated_annealing_{optimization_function.__name__}.gif', writer='pillow', fps=1)
    # plt.show()
