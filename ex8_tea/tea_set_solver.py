import os

import numpy as np
from matplotlib import pyplot as plt, animation

from ex8_tea.tea_set_calculator import SetCalculator


def save_image(calculator, filename):
    frame = calculator.calculate_fractal()
    colored_frame = calculator.render_frame(frame)

    plt.figure(figsize=(calculator.x_size / 100, calculator.y_size / 100), dpi=100)
    plt.imshow(colored_frame)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def create_animation(calculator, filename, frames=100):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fig = plt.figure(figsize=(calculator.x_size / 100, calculator.y_size / 100), dpi=100)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    img = ax.imshow(np.zeros((calculator.y_size, calculator.x_size, 3), dtype=np.uint8))

    def update(frame_num):
        frame = calculator.step()
        colored_frame = calculator.render_frame(frame)
        img.set_data(colored_frame)
        return [img]

    anim = animation.FuncAnimation(
        fig, update, frames=frames, interval=200, blit=True
    )
    anim.save(filename, writer='pillow', fps=10)
    plt.close()


def ts_solve():
    os.makedirs('Outputs', exist_ok=True)

    image_res_x = 1920
    image_res_y = 1080

    # image_res_x = 7680
    # image_res_y = 4320

    gif_res_x = 640
    gif_res_y = 360

    mandelbrot_points = [
        (-0.743643887037151, 0.131825904205330),
    ]

    julia_c_values = [
        (-0.7, 0.27015),
        (-0.8, 0.156),
        (-0.4, 0.6),
        (0.285, 0.01),
        (0.355, 0.355)
    ]

    print("Creating Mandelbrot image...")
    mandelbrot_calc = SetCalculator(
        fractal_type="mandelbrot",
        x_size=image_res_x,
        y_size=image_res_y,
        mandelbrot_point=mandelbrot_points[0]
    )
    save_image(mandelbrot_calc, f'Outputs/mandelbrot_image.png')

    print("Creating Julia images...")
    for i, c_value in enumerate(julia_c_values):
        julia_calc = SetCalculator(
            fractal_type="julia",
            x_size=image_res_x,
            y_size=image_res_y,
            julia_c_value=complex(*c_value)
        )
        save_image(julia_calc, f'Outputs/julia_image_{i + 1}.png')

    print("Creating Mandelbrot animation...")
    mandelbrot_calc = SetCalculator(
        fractal_type="mandelbrot",
        x_size=gif_res_x,
        y_size=gif_res_y,
        mandelbrot_point=mandelbrot_points[0]
    )
    create_animation(mandelbrot_calc, f'Outputs/mandelbrot_animation.gif', frames=100)

    print("Creating Julia animation...")
    julia_calc = SetCalculator(
        fractal_type="julia",
        x_size=gif_res_x,
        y_size=gif_res_y,
        julia_c_value=complex(*julia_c_values[1])
    )
    create_animation(julia_calc, f'Outputs/julia_animation.gif', frames=80)
