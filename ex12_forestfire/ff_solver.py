from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from ex12_forestfire.forest_fire import ForestFire


def ff_solve_pygame():
    import pygame
    pygame.init()

    neighb = "von_neumann"
    # neighb = "moore"
    start_burning = False
    p = 0.05
    f = 0.001
    density = 0.5
    width, height = 100, 100
    cell_size = 6
    fps = 10

    forest_fire = ForestFire(width=width, height=height, p=p, f=f, density=density, neighborhood=neighb,
                             start_burning=start_burning)

    # pygame animace

    screen_width = width * cell_size
    screen_height = height * cell_size
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Forest Fire")

    colors = [
        (80, 60, 40),  # NONE
        (0, 100, 0),  # TREE
        (255, 140, 0),  # BURNING
        (0, 0, 0)  # BURNED
    ]

    running = True
    clock = pygame.time.Clock()
    frame = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        forest_fire.step()
        frame += 1

        for y in range(height):
            for x in range(width):
                state = int(forest_fire.grid[y][x])
                pygame.draw.rect(screen, colors[state],
                                 (x * cell_size, y * cell_size, cell_size, cell_size))

        pygame.display.flip()
        pygame.display.set_caption(f"Forest Fire - Step {frame}")

        clock.tick(fps)

    pygame.quit()


def ff_solve_matplotlib():
    num_frames = 100

    neighb = "von_neumann"
    # neighb = "moore"
    start_burning = True
    p = 0.05
    f = 0.001
    density = 0.5
    width, height = 100, 100

    forest_fire = ForestFire(width=width, height=height, p=p, f=f,
                             density=density, neighborhood=neighb,
                             start_burning=start_burning)

    cmap = plt.cm.colors.ListedColormap([(0.31, 0.23, 0.16),  # NONE
                                         (0, 0.39, 0),  # TREE
                                         (1, 0.55, 0),  # BURNING
                                         (0, 0, 0)])  # BURNED
    bounds = [0, 1, 2, 3, 4]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title('Forest Fire - Step 0')
    ax.set_xticks([])
    ax.set_yticks([])

    img = ax.imshow(forest_fire.grid, cmap=cmap, norm=norm, interpolation='nearest')

    def update(frame):
        forest_fire.step()
        img.set_array(forest_fire.grid)
        ax.set_title(f'Forest Fire - Step {frame + 1}')
        return [img]

    animation = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True)

    print(f"Saving gif...")
    animation.save(f'Outputs/forest_fire.gif', writer='pillow', fps=10, dpi=100)

    plt.close(fig)
