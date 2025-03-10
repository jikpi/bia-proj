from typing import Type, Optional
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from function import OptimizationFunction


# Funkce pro vykresleni grafu, pro mnoho reseni podporuje pouze jednoduchou animaci
def graph_3d_generic(func: Type[OptimizationFunction], points: int = 100,
                     solutions: Optional[np.ndarray] = None,
                     save_result: bool = False,
                     save_gif: bool = False,
                     save_single_rotating: bool = False,
                     frames: int = 360,
                     input_text: Optional[str] = None):
    # Vytvoreni prostoru v doporucenem rozmezi
    x_range, y_range = func.recommended_range(), func.recommended_range()
    x = np.linspace(x_range[0], x_range[1], points)
    y = np.linspace(y_range[0], y_range[1], points)
    X, Y = np.meshgrid(x, y)

    # Evaluace
    Z = np.zeros_like(X)
    for i in range(points):
        for j in range(points):
            Z[i, j] = func.evaluate(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot
    ax.set_facecolor('#050505')  # Pozadi
    fig.patch.set_facecolor('#050505')  # Fig

    # Osa
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('purple')
    ax.yaxis.pane.set_edgecolor('purple')
    ax.zaxis.pane.set_edgecolor('purple')
    ax.grid(True, color='purple', linestyle=':', alpha=0.5)

    # Text osy
    ax.set_xlabel('X', color='white', fontweight='bold')
    ax.set_ylabel('Y', color='white', fontweight='bold')
    ax.set_zlabel('Z', color='white', fontweight='bold')
    ax.set_title(f'Funkce: {func.__name__}', color='#4cc9f0', fontweight='bold', fontsize=16)

    # Barvy tick
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    # Plocha
    surface = ax.plot_surface(X, Y, Z, cmap='winter', antialiased=True, alpha=0.8, zorder=1)

    # Contour cary
    contour = ax.contour(X, Y, Z, zdir='z', offset=Z.min(), cmap='winter', alpha=0.5)

    # Napoveda barev
    cbar = fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.1)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Uhel pohledu
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()

    # Zvoleny text
    if input_text:
        ax.text2D(0.05, 0.95, input_text, transform=ax.transAxes, color='white', fontweight='bold', fontsize=12)

    # Logika vykresleni
    def plot_solution(solution):
        solution_z = func.evaluate(solution)

        # Cary napoveda
        ax.plot([solution[0], solution[0]], [solution[1], solution[1]], [Z.min(), solution_z],
                color='red', linestyle='--', linewidth=2, zorder=10)
        ax.plot([solution[0], solution[0]], [y_range[0], solution[1]], [Z.min(), Z.min()],
                color='red', linestyle='--', linewidth=2, zorder=10)
        ax.plot([x_range[0], solution[0]], [solution[1], solution[1]], [Z.min(), Z.min()],
                color='red', linestyle='--', linewidth=2, zorder=10)

        # Reseni
        ax.scatter(solution[0], solution[1], solution_z, color='red', s=200, marker='*',
                   label='Solution', zorder=11, edgecolors='white', linewidth=1.5)

        # Text k reseni
        ax.text(solution[0], solution[1], solution_z, f'({solution[0]:.2f}, {solution[1]:.2f}, {solution_z:.2f})',
                color='white', fontweight='bold', fontsize=10, zorder=12)

    if solutions is not None:
        solutions = np.atleast_2d(solutions)
        if solutions.shape[0] == 1 or not save_gif:
            plot_solution(solutions[-1])
        elif solutions.shape[0] > 1 and save_result and save_gif:
            def update(frame):
                ax.clear()
                ax.plot_surface(X, Y, Z, cmap='winter', antialiased=True, alpha=0.8, zorder=1)
                plot_solution(solutions[frame % solutions.shape[0]])
                ax.view_init(elev=20, azim=45)
                if input_text:
                    ax.text2D(0.05, 0.95, input_text, transform=ax.transAxes, color='white', fontweight='bold',
                              fontsize=12)
                return fig,

            ani = FuncAnimation(fig, update, frames=solutions.shape[0], interval=500, blit=True)
            ani.save(f'Outputs/gen_{func.__name__}_solutions.gif', writer='pillow', fps=2)
            print(f'Saved gif with multiple solutions for {func.__name__}')

    if save_result:
        if save_single_rotating:
            def update(frame):
                ax.view_init(elev=20, azim=frame * (360 / frames))
                return fig,

            print(f'Saving rotating gif for {func.__name__}')
            ani = FuncAnimation(fig, update, frames=frames, interval=16.67, blit=True)
            ani.save(f'Outputs/gen_{func.__name__}_rotation.gif', writer='pillow', fps=60)
            print('Done')
        else:
            plt.savefig(f'Outputs/gen_{func.__name__}_surface.png')
            print(f'Saved static image for {func.__name__}')

    plt.show()
