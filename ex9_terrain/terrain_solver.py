import os

import numpy as np
from matplotlib import pyplot as plt, animation
from matplotlib.colors import LinearSegmentedColormap

from ex9_terrain.terrain_generator import Terrain


def terrain_animation(terrain, num_steps=5, sea_rise_fps=10, terrain_fps=1,
                      sea_rise_frames=100, pause_seconds=3, save_gif=True):
    fig = plt.figure(figsize=(18, 8), facecolor='black')
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    for ax in [ax1, ax2]:
        ax.set_facecolor('black')
        ax.set_xticks([])
        ax.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax2.set_zticks([])
    ax2.view_init(elev=30, azim=45)
    ax2.grid(False)

    colors = [(0.0, 'darkblue'), (0.3, 'green'), (0.5, 'yellowgreen'),
              (0.7, 'sandybrown'), (0.85, 'gray'), (1.0, 'white')]
    terrain_cmap = LinearSegmentedColormap.from_list('terrain_cmap', colors)
    water_color = 'royalblue'
    water_cmap = LinearSegmentedColormap.from_list('water_cmap', [(0.0, water_color), (1.0, water_color)])

    print("Generating terrain...")
    heights = terrain.heights.copy()
    all_heights = [heights.copy()]
    for i in range(num_steps):
        heights = terrain.step()
        all_heights.append(heights.copy())
    print("Creating animation...")

    min_heights = min(h.min() for h in all_heights)
    max_heights = max(h.max() for h in all_heights)
    final_heights = all_heights[-1]

    height_range = max_heights - min_heights
    if height_range == 0: height_range = 1.0
    min_h_plot = min_heights - height_range * 0.05
    max_h_plot = max_heights + height_range * 0.05

    title = fig.suptitle("Initializing...", color='white', fontsize=16)

    num_terrain_steps = len(all_heights)
    effective_terrain_fps = max(terrain_fps, 1e-6)
    terrain_frame_multiplier = max(1, round(sea_rise_fps / effective_terrain_fps))
    num_terrain_display_frames = num_terrain_steps * terrain_frame_multiplier

    num_pause_frames = max(0, round(pause_seconds * sea_rise_fps))
    total_animation_frames = num_terrain_display_frames + sea_rise_frames
    total_frames_with_pause = total_animation_frames + num_pause_frames

    sea_start = min_h_plot + height_range * 0.1
    sea_end = min_h_plot + height_range * 0.5
    sea_levels = np.linspace(sea_start, sea_end, sea_rise_frames)

    plotted_artists = []

    def update(frame):
        nonlocal title

        is_pause_frame = frame >= total_animation_frames

        if frame % 20 == 0:
            print(f"Animating frame {frame + 1}/{total_frames_with_pause}")

        ax1.clear()
        ax2.clear()
        ax1.set_facecolor('black')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.set_aspect('equal')

        ax2.set_facecolor('black')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])
        ax2.set_zlim(min_h_plot, max_h_plot)
        ax2.view_init(elev=30, azim=45)
        ax2.grid(False)

        if is_pause_frame:
            current_heights = final_heights
            sea_level = sea_levels[-1]
            water_mask = current_heights < sea_level
            is_sea_phase = True
            frame_title = f"Final Frame (Sea Level: {sea_level:.2f})"
        elif frame < num_terrain_display_frames:
            terrain_step_index = frame // terrain_frame_multiplier
            terrain_step_index = min(terrain_step_index, num_terrain_steps - 1)
            current_heights = all_heights[terrain_step_index]
            is_sea_phase = False
            frame_title = f"Terrain Generation: Step {terrain_step_index}"
            water_mask = None
        else:
            current_heights = final_heights
            sea_frame_index = frame - num_terrain_display_frames
            sea_frame_index = min(sea_frame_index, sea_rise_frames - 1)
            sea_level = sea_levels[sea_frame_index]
            water_mask = current_heights < sea_level
            is_sea_phase = True
            frame_title = f"Sea Level Rising: {sea_level:.2f}"

        y_size, x_size = current_heights.shape
        x = np.arange(0, x_size, 1)
        y = np.arange(0, y_size, 1)
        X, Y = np.meshgrid(x, y)
        downsample = max(1, x_size // 100, y_size // 100)

        current_artists = []

        if is_sea_phase:
            masked_terrain = np.ma.masked_array(current_heights, mask=water_mask)
            terrain_img = ax1.imshow(masked_terrain, cmap=terrain_cmap, vmin=min_h_plot, vmax=max_h_plot,
                                     interpolation='nearest', origin='lower', extent=[0, x_size, 0, y_size])
            water_display_data = np.ma.masked_array(np.full(current_heights.shape, sea_level), mask=~water_mask)
            water_img = ax1.imshow(water_display_data, cmap=water_cmap, vmin=min_h_plot, vmax=max_h_plot,
                                   interpolation='nearest', origin='lower', extent=[0, x_size, 0, y_size])
            current_artists.extend([terrain_img, water_img])
        else:
            img = ax1.imshow(current_heights, cmap=terrain_cmap, vmin=min_h_plot, vmax=max_h_plot,
                             interpolation='nearest', origin='lower', extent=[0, x_size, 0, y_size])
            current_artists.append(img)
        ax1.set_xlim(0, x_size)
        ax1.set_ylim(0, y_size)

        X_ds = X[::downsample, ::downsample]
        Y_ds = Y[::downsample, ::downsample]
        Z_ds = current_heights[::downsample, ::downsample]

        terrain_surf = ax2.plot_surface(X_ds, Y_ds, Z_ds, cmap=terrain_cmap,
                                        vmin=min_h_plot, vmax=max_h_plot,
                                        rstride=1, cstride=1, linewidth=0, antialiased=True,
                                        alpha=0.95 if is_sea_phase else 1.0)
        current_artists.append(terrain_surf)
        ax2.set_xlim(0, x_size)
        ax2.set_ylim(0, y_size)

        if is_sea_phase:
            water_surface_heights = np.full(current_heights.shape, sea_level)
            water_surface_heights = np.where(water_mask, water_surface_heights, np.nan)
            water_z_offset = 1e-5
            water_surf = ax2.plot_surface(X_ds, Y_ds,
                                          water_surface_heights[::downsample, ::downsample] + water_z_offset,
                                          color=water_color, alpha=0.75,
                                          rstride=1, cstride=1, linewidth=0, antialiased=True)
            current_artists.append(water_surf)

        title.set_text(frame_title)
        current_artists.append(title)

        plotted_artists[:] = current_artists
        return plotted_artists

    anim_interval = 1000 / sea_rise_fps
    anim = animation.FuncAnimation(fig, update, frames=total_frames_with_pause, interval=anim_interval,
                                   blit=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_gif:
        output_dir = "Outputs"
        os.makedirs(output_dir, exist_ok=True)
        seed_str = f"_seed{terrain.seed}" if hasattr(terrain, "seed") and terrain.seed is not None else ""
        filename = os.path.join(output_dir, f"terrain_{seed_str}.gif")
        try:
            writer = animation.PillowWriter(fps=sea_rise_fps)
            anim.save(filename, writer=writer)
            print(f"Animation saved successfully.")
        except Exception as e:
            import traceback
            traceback.print_exc()


def create_three_terrains():
    fig = plt.figure(figsize=(12, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)

    color_sets = [
        [(0.0, 'darkblue'), (0.5, 'royalblue'), (1.0, 'lightskyblue')],
        [(0.0, 'darkgreen'), (0.5, 'forestgreen'), (1.0, 'limegreen')],
        [(0.0, 'firebrick'), (0.5, 'orangered'), (1.0, 'gold')]
    ]

    cmaps = [LinearSegmentedColormap.from_list(f'terrain_cmap_{i}', colors) for i, colors in enumerate(color_sets)]

    terrains = [
        Terrain(pt_init=1.0, pt_damping=0.7, seed=11),
        Terrain(pt_init=1.0, pt_damping=0.9, seed=22),
        Terrain(pt_init=1.0, pt_damping=0.99, seed=33)
    ]

    for terrain in terrains:
        for _ in range(6):
            terrain.step()

    offsets = [0, 5, 10]

    heights_all = []
    for terrain in terrains:
        heights_all.append(terrain.heights)

    y_size, x_size = heights_all[0].shape
    downsample = max(1, x_size // 100, y_size // 100)

    x = np.arange(0, x_size, 1)
    y = np.arange(0, y_size, 1)
    X, Y = np.meshgrid(x, y)
    X_ds = X[::downsample, ::downsample]
    Y_ds = Y[::downsample, ::downsample]

    for i, (heights, cmap, offset) in enumerate(zip(heights_all, cmaps, offsets)):
        Z_ds = heights[::downsample, ::downsample] + offset

        min_h = heights.min()
        max_h = heights.max()
        norm_heights = (heights - min_h) / (max_h - min_h) if max_h > min_h else heights

        surf = ax.plot_surface(X_ds, Y_ds, Z_ds,
                               cmap=cmap,
                               rstride=1, cstride=1,
                               linewidth=0,
                               antialiased=True,
                               alpha=0.9)

    ax.view_init(elev=30, azim=45)

    ax.set_xlim(0, x_size)
    ax.set_ylim(0, y_size)

    plt.title("Multi terrain", color='white', fontsize=16, pad=20)
    plt.tight_layout()

    plt.savefig("Outputs/multi_terrain.png", dpi=300, bbox_inches='tight')


def terrain_solve():
    os.makedirs("Outputs", exist_ok=True)

    # pro splneni zadani, ale animace nize je o hodne zajimavejsi
    print("# Creating 3 terrains...")
    create_three_terrains()
    print("# Creating terrain animation...")

    terrain = Terrain(pt_init=1.0, pt_damping=0.5, seed=10)
    terrain_animation(terrain, 6, 10, 1, 70, 4, True)
