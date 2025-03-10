import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from BIA.ex_4_ga_tsp.ga_tsp_solver import TspGaGenerationData


def create_city_path_animation(generations_data: list[TspGaGenerationData],
                               fig_size=(12, 8),
                               animation_interval=500,
                               city_marker_size=100,
                               path_color='blue',
                               city_color='red'):
    cities = generations_data[0].entities[0].city_list

    x_coords = [city.x for city in cities]
    y_coords = [city.y for city in cities]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    padding = 0.1 * max(x_max - x_min, y_max - y_min)

    fig, ax = plt.subplots(figsize=fig_size)

    def init():
        ax.clear()
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)

        for city in cities:
            ax.scatter(city.x, city.y, c=city_color, s=city_marker_size)
            ax.annotate(f'{city.id}', (city.x, city.y),
                        xytext=(5, 5), textcoords='offset points')

        ax.set_title('City Path Evolution')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        return []

    def animate(frame):
        init()

        generation_idx = frame // len(cities)
        step = frame % len(cities)

        if generation_idx >= len(generations_data):
            return []

        generation = generations_data[generation_idx]
        best_entity = generation.best_entity

        lines = []
        for i in range(step):
            city1 = best_entity.city_list[i]
            city2 = best_entity.city_list[i + 1]
            line, = ax.plot([city1.x, city2.x], [city1.y, city2.y],
                            c=path_color, linewidth=1.5)
            lines.append(line)

        if step == len(cities) - 1:
            city1 = best_entity.city_list[-1]
            city2 = best_entity.city_list[0]
            line, = ax.plot([city1.x, city2.x], [city1.y, city2.y],
                            c=path_color, linewidth=1.5)
            lines.append(line)

        ax.text(0.02, 0.98, f'Generation: {generation.generation}',
                transform=ax.transAxes, verticalalignment='top')
        ax.text(0.02, 0.94, f'Distance: {generation.shortest_distance:.2f}',
                transform=ax.transAxes, verticalalignment='top')

        return lines

    total_frames = len(generations_data) * len(cities)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=total_frames, interval=animation_interval,
                                   blit=True)

    anim.save('Outputs/city_path_evolution.gif', writer='pillow')


from typing import List


def visualize_city_evolution(generation_data: List[TspGaGenerationData],
                             x_size: float = 1000,
                             y_size: float = 1000,
                             interval: int = 1000,
                             final_frame_duration: int = 3000,
                             city_seed: int = None,
                             solution_seed: int = None,
                             city_size: int = None):
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['#4ECDC4', '#FF6B6B']
    alphas = [1.0, 0.3]

    significant_generations = []
    prev_distance = float('inf')

    for gen in generation_data:
        current_distance = gen.shortest_distance
        if abs(current_distance - prev_distance) > 1e-10:
            significant_generations.append(gen)
            prev_distance = current_distance

    def init():
        ax.clear()
        ax.set_xlim(0, x_size)
        ax.set_ylim(0, y_size)
        return []

    def animate(frame):
        ax.clear()

        ax.set_xlim(0, x_size)
        ax.set_ylim(0, y_size)

        total_normal_frames = len(significant_generations) * 2

        if frame >= total_normal_frames:
            actual_generation_index = len(significant_generations) - 1
            is_comparison_frame = False
        else:
            is_comparison_frame = frame % 2 == 0
            actual_generation_index = frame // 2

        title = f'Generation {significant_generations[actual_generation_index].generation}\n'
        city_seed_text = str(city_seed) if city_seed is not None else "None"
        solution_seed_text = str(solution_seed) if solution_seed is not None else "None"
        title += f'City Seed: {city_seed_text} | Solution Seed: {solution_seed_text}'

        ax.set_title(title)

        cities = significant_generations[actual_generation_index].best_entity.city_list

        for city in cities:
            ax.scatter(city.x, city.y, c='black', s=100, zorder=3)
            ax.annotate(f'{city.id}', (city.x + 1, city.y + 1), zorder=3)

        if is_comparison_frame and actual_generation_index > 0:
            for hist_idx in range(2):
                if actual_generation_index - hist_idx >= 0:
                    gen = significant_generations[actual_generation_index - hist_idx]
                    cities = gen.best_entity.city_list

                    coords = np.array([(city.x, city.y) for city in cities])
                    coords = np.vstack([coords, coords[0]])

                    ax.plot(coords[:, 0], coords[:, 1],
                            color=colors[hist_idx],
                            alpha=alphas[hist_idx],
                            linewidth=2,
                            zorder=2 - hist_idx,
                            label=f'Gen {gen.generation}')
        else:
            gen = significant_generations[actual_generation_index]
            cities = gen.best_entity.city_list

            coords = np.array([(city.x, city.y) for city in cities])
            coords = np.vstack([coords, coords[0]])

            ax.plot(coords[:, 0], coords[:, 1],
                    color=colors[0],
                    alpha=alphas[0],
                    linewidth=2,
                    zorder=2,
                    label=f'Gen {gen.generation}')

        ax.legend(loc='upper right')

        current_distance = significant_generations[actual_generation_index].shortest_distance
        ax.text(0.02, 0.98, f'Distance: {current_distance:.2f}',
                transform=ax.transAxes,
                verticalalignment='top')

        return []

    total_normal_frames = len(significant_generations) * 2
    extra_frames = final_frame_duration // interval
    total_frames = total_normal_frames + extra_frames

    anim = animation.FuncAnimation(fig,
                                   animate,
                                   init_func=init,
                                   frames=total_frames,
                                   interval=interval,
                                   blit=True)

    anim.save(f'Outputs/ga_tsp-c{city_seed}-s{solution_seed}-np{city_size}.gif', writer='pillow')

    plt.close()
    return anim
