from typing import Type, List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from ex_7_soma.soma_progress_repr import MigrationSnapshot
from ex_7_soma.soma_repr import Population
from function import OptimizationFunction


def graph_soma_optimization(optimization_function: Type['OptimizationFunction'],
                            population_history: List['Population']):
    print(f"Creating animation for {optimization_function.__name__}...")
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    ax3d = fig.add_subplot(gs[0], projection='3d')
    ax2d = fig.add_subplot(gs[1])

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

    ax3d.set_xlabel('X', color='white', fontweight='bold')
    ax3d.set_ylabel('Y', color='white', fontweight='bold')
    ax3d.set_zlabel('Z', color='white', fontweight='bold')
    ax3d.set_title(f'3D View: {optimization_function.__name__}', color='#4cc9f0', fontweight='bold', fontsize=16)

    ax2d.grid(True, color='purple', linestyle=':', alpha=0.5)
    ax2d.set_xlabel('X', color='white', fontweight='bold')
    ax2d.set_ylabel('Y', color='white', fontweight='bold')
    ax2d.set_title(f'2D View: {optimization_function.__name__}', color='#4cc9f0', fontweight='bold', fontsize=16)

    for ax in [ax3d, ax2d]:
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
    ax3d.tick_params(axis='z', colors='white')

    x_range, y_range = optimization_function.recommended_range(), optimization_function.recommended_range()
    points = 100
    x = np.linspace(x_range[0], x_range[1], points)
    y = np.linspace(y_range[0], y_range[1], points)
    X, Y = np.meshgrid(x, y)

    Z = np.array(
        [[optimization_function.evaluate(np.array([X[i, j], Y[i, j]])) for j in range(points)] for i in range(points)])

    surface = ax3d.plot_surface(X, Y, Z, cmap='winter', antialiased=True, alpha=0.5)
    contour = ax2d.contourf(X, Y, Z, levels=20, cmap='winter', alpha=0.5)

    cbar = fig.colorbar(surface, ax=ax3d, shrink=0.5, aspect=10, pad=0.1)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    individuals_3d = ax3d.scatter([], [], [], color='yellow', s=50, label='Individuals')
    leader_3d = ax3d.scatter([], [], [], color='red', s=100, label='Leader')

    individuals_2d = ax2d.scatter([], [], color='yellow', s=50, label='Individuals')
    leader_2d = ax2d.scatter([], [], color='red', s=100, label='Leader')

    ax3d.legend(loc='upper right', facecolor='#050505', edgecolor='purple', labelcolor='white')
    ax2d.legend(loc='upper right', facecolor='#050505', edgecolor='purple', labelcolor='white')

    best_solution_text = fig.text(0.5, 0.02, '', ha='center', va='center', color='white', fontsize=12)

    def update(frame):
        population = population_history[frame]

        x_coords = [individual.params[0] for individual in population.individuals]
        y_coords = [individual.params[1] for individual in population.individuals]
        z_coords = [individual.eval(optimization_function) for individual in population.individuals]

        individuals_3d._offsets3d = (x_coords, y_coords, z_coords)
        individuals_2d.set_offsets(np.column_stack((x_coords, y_coords)))

        if population.leader is not None:
            leader = population.individuals[population.leader]
            leader_value = leader.eval(optimization_function)

            leader_3d._offsets3d = ([leader.params[0]],
                                    [leader.params[1]],
                                    [leader_value])
            leader_2d.set_offsets(np.array([leader.params[0],
                                            leader.params[1]]).reshape(1, 2))

            best_solution_text.set_text(
                f"Generation {frame}: Leader f({leader.params[0]:.2f}, "
                f"{leader.params[1]:.2f}) = {leader_value:.2f}")

        return individuals_3d, leader_3d, individuals_2d, leader_2d, best_solution_text

    plt.tight_layout()

    anim = FuncAnimation(fig, update, frames=len(population_history),
                         interval=1000, blit=False, repeat=True)
    anim.save(f'Outputs/soma_optimization_{optimization_function.__name__}.gif',
              writer='pillow', fps=1)


def graph_soma_optimization_complex(optimization_function: Type['OptimizationFunction'],
                                    migration_history: List['MigrationSnapshot']):
    print(f"Creating animation for {optimization_function.__name__}...")
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()

    fig.patch.set_facecolor('#050505')
    ax.set_facecolor('#050505')
    ax.grid(True, color='purple', linestyle=':', alpha=0.5)
    ax.set_xlabel('X', color='white', fontweight='bold')
    ax.set_ylabel('Y', color='white', fontweight='bold')
    ax.set_title(f'SOMA Optimization: {optimization_function.__name__}',
                 color='#4cc9f0', fontweight='bold', fontsize=16)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    x_range, y_range = optimization_function.recommended_range(), optimization_function.recommended_range()
    points = 100
    x = np.linspace(x_range[0], x_range[1], points)
    y = np.linspace(y_range[0], y_range[1], points)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[optimization_function.evaluate(np.array([X[i, j], Y[i, j]]))
                   for j in range(points)] for i in range(points)])

    contour = ax.contourf(X, Y, Z, levels=20, cmap='winter', alpha=0.5)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    individuals = ax.scatter([], [], color='yellow', s=100, label='Individuals', zorder=3)
    leaders = ax.scatter([], [], color='red', s=150, marker='*', label='Leaders', zorder=4)
    active_individual = ax.scatter([], [], color='cyan', s=120, label='Active Individual', zorder=5)
    path_current = ax.plot([], [], 'cyan', alpha=0.7, linewidth=2, label='Current Path', zorder=2)[0]
    path_history = ax.scatter([], [], color='white', s=10, alpha=0.2, label='Path History', zorder=1)

    ax.legend(loc='upper right', facecolor='#050505', edgecolor='purple', labelcolor='white')

    info_text = fig.text(0.02, 0.02, '', color='white', fontsize=12, ha='left', va='bottom')

    def create_animation_frames():
        frames = []
        for migration_idx, snapshot in enumerate(migration_history):
            initial_frame = {
                'migration': migration_idx,
                'individual_idx': None,
                'leader_idx': None,
                'positions': [(traj.start_position[0], traj.start_position[1])
                              for traj in snapshot.trajectories if traj is not None],
                'path_points': [],
                'current_path': [],
                'active_pos': None,
                'leader_pos': None,
                'phase': 'start',
                'best_pos': snapshot.best_position,
                'best_val': snapshot.best_fitness
            }
            frames.append(initial_frame)

            for ind_idx, trajectory in enumerate(snapshot.trajectories):
                if trajectory is None:
                    continue

                current_positions = []
                for i, traj in enumerate(snapshot.trajectories):
                    if traj is None:
                        continue
                    if i < ind_idx:
                        current_positions.append((traj.final_position[0], traj.final_position[1]))
                    else:
                        current_positions.append((traj.start_position[0], traj.start_position[1]))

                for leader_idx, path_points in trajectory.paths.items():
                    path_coords = [(p.position[0], p.position[1]) for p in path_points]

                    frame = {
                        'migration': migration_idx,
                        'individual_idx': ind_idx,
                        'leader_idx': leader_idx,
                        'positions': current_positions,
                        'path_points': [],
                        'current_path': path_coords,
                        'active_pos': current_positions[ind_idx],
                        'leader_pos': current_positions[leader_idx],
                        'phase': 'moving',
                        'best_pos': snapshot.best_position,
                        'best_val': snapshot.best_fitness
                    }
                    frames.append(frame)

            final_frame = {
                'migration': migration_idx,
                'individual_idx': None,
                'leader_idx': None,
                'positions': [(traj.final_position[0], traj.final_position[1])
                              for traj in snapshot.trajectories if traj is not None],
                'path_points': [(p.position[0], p.position[1])
                                for traj in snapshot.trajectories if traj is not None
                                for path in traj.paths.values()
                                for p in path],
                'current_path': [],
                'active_pos': None,
                'leader_pos': None,
                'phase': 'end',
                'best_pos': snapshot.best_position,
                'best_val': snapshot.best_fitness
            }
            frames.append(final_frame)

        return frames

    def update(frame_data):
        x_coords = [pos[0] for pos in frame_data['positions']]
        y_coords = [pos[1] for pos in frame_data['positions']]
        individuals.set_offsets(np.column_stack([x_coords, y_coords]))

        if frame_data['active_pos'] is not None:
            active_individual.set_offsets([frame_data['active_pos']])
            active_individual.set_sizes([120])
        else:
            active_individual.set_sizes([0])

        if frame_data['leader_pos'] is not None:
            leaders.set_offsets([frame_data['leader_pos']])
            leaders.set_sizes([150])
        else:
            leaders.set_sizes([0])

        if frame_data['current_path']:
            path_x = [pos[0] for pos in frame_data['current_path']]
            path_y = [pos[1] for pos in frame_data['current_path']]
            path_current.set_data(path_x, path_y)
        else:
            path_current.set_data([], [])

        if frame_data['path_points']:
            history_x = [pos[0] for pos in frame_data['path_points']]
            history_y = [pos[1] for pos in frame_data['path_points']]
            path_history.set_offsets(np.column_stack([history_x, history_y]))
            path_history.set_sizes([10] * len(history_x))
        else:
            path_history.set_sizes([0])

        phase_text = ''
        if frame_data['phase'] == 'start':
            phase_text = 'Starting positions'
        elif frame_data['phase'] == 'moving':
            phase_text = f"Individual {frame_data['individual_idx']} following leader {frame_data['leader_idx']}"
        elif frame_data['phase'] == 'end':
            phase_text = 'Final positions'

        info_text.set_text(
            f"Migration {frame_data['migration']} - {phase_text}\n"
            f"Best value: {frame_data['best_val']:.4f}"
        )

        return individuals, leaders, active_individual, path_current, path_history, info_text

    frames = create_animation_frames()
    anim = FuncAnimation(fig, update, frames=frames,
                         interval=500, blit=True, repeat=True)

    plt.tight_layout()
    anim.save(f'Outputs/soma_optimization_{optimization_function.__name__}-complex.gif',
              writer='pillow', fps=2)
