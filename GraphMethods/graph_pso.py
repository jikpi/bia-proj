import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import Type, List

from ex_6_pso.particle_repr import Swarm
from function import OptimizationFunction


def graph_pso_optimization(optimization_function: Type['OptimizationFunction'],
                         swarm_history: List['Swarm'],
                         init_seed: int = None,
                         solution_seed: int = None):
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

    particles_3d = ax3d.scatter([], [], [], color='yellow', s=50, label='Particles')
    best_3d = ax3d.scatter([], [], [], color='red', s=100, label='Global Best')

    particles_2d = ax2d.scatter([], [], color='yellow', s=50, label='Particles')
    best_2d = ax2d.scatter([], [], color='red', s=100, label='Global Best')

    ax3d.legend(loc='upper right', facecolor='#050505', edgecolor='purple', labelcolor='white')
    ax2d.legend(loc='upper right', facecolor='#050505', edgecolor='purple', labelcolor='white')

    best_solution_text = fig.text(0.5, 0.02, '', ha='center', va='center', color='white', fontsize=12)

    def update(frame):
        swarm = swarm_history[frame]

        x_coords = [particle.params[0] for particle in swarm.particles]
        y_coords = [particle.params[1] for particle in swarm.particles]
        z_coords = [particle.eval(optimization_function) for particle in swarm.particles]

        particles_3d._offsets3d = (x_coords, y_coords, z_coords)
        particles_2d.set_offsets(np.column_stack((x_coords, y_coords)))

        best_value = swarm.best_particle.eval(optimization_function)
        best_3d._offsets3d = ([swarm.best_particle.params[0]], [swarm.best_particle.params[1]], [best_value])
        best_2d.set_offsets(np.array([swarm.best_particle.params[0], swarm.best_particle.params[1]]).reshape(1, 2))

        best_solution_text.set_text(
            f"Generation {frame}: Best f({swarm.best_particle.params[0]:.2f}, {swarm.best_particle.params[1]:.2f}) = {best_value:.2f}")

        return particles_3d, best_3d, particles_2d, best_2d, best_solution_text

    plt.tight_layout()

    anim = FuncAnimation(fig, update, frames=len(swarm_history), interval=1000, blit=False, repeat=True)
    anim.save(f'Outputs/pso_optimization_is{init_seed}_ss{solution_seed}-{optimization_function.__name__}.gif',
             writer='pillow', fps=1)