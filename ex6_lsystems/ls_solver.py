import math
import os

from matplotlib import pyplot as plt, animation
from matplotlib.animation import FuncAnimation

from ex6_lsystems.l_system import LSystem


def visualize_lsystem(lsystem: LSystem, expansions, step_length=10, filename=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    all_expansions = [lsystem.axiom]

    for i in range(expansions):
        lsystem.expand()
        all_expansions.append(lsystem.get_string())

    def draw_lsystem(lsystem_string, angle):
        x, y = 0, 0
        direction = 90

        stack = []
        lines_x, lines_y = [], []
        current_line_x, current_line_y = [x], [y]

        for char in lsystem_string:
            if char.isupper():
                # krok s carou
                new_x = x + step_length * math.cos(math.radians(direction))
                new_y = y + step_length * math.sin(math.radians(direction))
                x, y = new_x, new_y
                current_line_x.append(x)
                current_line_y.append(y)
            elif char == 'b':
                # krok bez cary
                x = x + step_length * math.cos(math.radians(direction))
                y = y + step_length * math.sin(math.radians(direction))
                lines_x.append(current_line_x)
                lines_y.append(current_line_y)
                current_line_x, current_line_y = [x], [y]
            elif char == '+':
                # otoceni se doprava
                direction -= angle
            elif char == '-':
                # otoceni se doleva
                direction += angle
            elif char == '[':
                # ulozeni momentalni pozice na zasobnik
                stack.append((x, y, direction))
            elif char == ']':
                # vraceni se na posledni ulozenou pozici
                if stack:
                    if current_line_x:
                        lines_x.append(current_line_x)
                        lines_y.append(current_line_y)

                    x, y, direction = stack.pop()
                    current_line_x, current_line_y = [x], [y]

        if current_line_x:
            lines_x.append(current_line_x)
            lines_y.append(current_line_y)

        return lines_x, lines_y

    def fit_plot(all_lines_x, all_lines_y):
        all_x = [x for lines_x in all_lines_x for line in lines_x for x in line]
        all_y = [y for lines_y in all_lines_y for line in lines_y for y in line]

        if not all_x or not all_y:
            return 0, 0, 1, 1

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        padding = max(max_x - min_x, max_y - min_y) * 0.1
        return min_x - padding, max_x + padding, min_y - padding, max_y + padding

    # stabilizace plotu (pro stabilni animaci expanzi)
    all_lines_x = []
    all_lines_y = []
    frames = []

    for gen_idx, gen_string in enumerate(all_expansions):
        lines_x, lines_y = draw_lsystem(gen_string, lsystem.angle)
        all_lines_x.append(lines_x)
        all_lines_y.append(lines_y)
        frames.append((gen_idx, lines_x, lines_y))

    # globalni min a max pro vsechny expanze
    global_min_x, global_max_x, global_min_y, global_max_y = fit_plot(all_lines_x, all_lines_y)

    center_x = (global_min_x + global_max_x) / 2
    center_y = (global_min_y + global_max_y) / 2

    width = global_max_x - global_min_x
    height = global_max_y - global_min_y

    max_dim = max(width, height) / 2

    def update(frame):
        gen_idx, lines_x, lines_y = frame
        ax.clear()
        ax.set_facecolor('white')

        for lx, ly in zip(lines_x, lines_y):
            ax.plot(lx, ly, 'k-', linewidth=1)

        ax.set_xlim(center_x - max_dim, center_x + max_dim)
        ax.set_ylim(center_y - max_dim, center_y + max_dim)
        ax.set_aspect('equal')
        ax.set_title(f'Expansion {gen_idx}', fontsize=14)
        ax.set_axis_off()

        return ax,

    ani = FuncAnimation(fig, update, frames=frames, blit=False)

    output_filename = filename if filename else "lsystem_animation.gif"
    output_filename = "LS_" + output_filename
    output_path = os.path.join("Outputs", output_filename)
    ani.save(output_path, writer=animation.PillowWriter(fps=1 / 2),
             dpi=100, savefig_kwargs={'facecolor': 'white'})

    plt.close(fig)

    return output_path


def ls_solve():
    # vsechny z prilozene stranky, odkomentovany nejzajimavejsi
    # (vsechny jsem je podrobne nekontroloval, tedy jako 'odevzdani' lze pocitat pouze ty nekomentovane)

    l_systems = [
        # {
        #     "name": "Koch_Curve",
        #     "axiom": "F",
        #     "rules": [("F", "F-F++F-F")],
        #     "angle": 60,
        #     "expansions": 4
        # },
        # {
        #     "name": "Koch_Snowflake",
        #     "axiom": "F++F++F",
        #     "rules": [("F", "F-F++F-F")],
        #     "angle": 60,
        #     "expansions": 4
        # },
        {
            "name": "Variant_Koch_Curve",
            "axiom": "F+F+F",
            "rules": [("F", "F-F+F")],
            "angle": 120,
            "expansions": 7
        },
        # {
        #     "name": "Square_Koch_Variant",
        #     "axiom": "F+F+F+F",
        #     "rules": [("F", "FF+F++F+F")],
        #     "angle": 90,
        #     "expansions": 3
        # },
        # {
        #     "name": "Dragon_Curve",
        #     "axiom": "FX",
        #     "rules": [("X", "X+YF+"), ("Y", "-FX-Y")],
        #     "angle": 90,
        #     "expansions": 10
        # },
        {
            "name": "Gosper_Curve",
            "axiom": "XF",
            "rules": [
                ("X", "X+YF++YF-FX--FXFX-YF+"),
                ("Y", "-FX+YFYF++YF+FX--FX-Y")
            ],
            "angle": 60,
            "expansions": 5
        },
        # {
        #     "name": "Sierpinski_Curve",
        #     "axiom": "F+XF+F+XF",
        #     "rules": [("X", "XF-F+F-XF+F+XF-F+F-X")],
        #     "angle": 90,
        #     "expansions": 3
        # },
        # {
        #     "name": "Gilbert_Curve",
        #     "axiom": "X",
        #     "rules": [
        #         ("X", "-YF+XFX+FY-"),
        #         ("Y", "+XF-YFY-FX+")
        #     ],
        #     "angle": 90,
        #     "expansions": 6
        # },
        # {
        #     "name": "Square_Curve_Variant_1",
        #     "axiom": "F+F+F+F",
        #     "rules": [("F", "FF+F+F+F+FF")],
        #     "angle": 90,
        #     "expansions": 3
        # },
        # {
        #     "name": "Square_Curve_Variant_2",
        #     "axiom": "F+F+F+F",
        #     "rules": [("F", "F+F-F-FF+F+F-F")],
        #     "angle": 90,
        #     "expansions": 3
        # },
        # {
        #     "name": "Koch_Generalization_1",
        #     "axiom": "F+F+F+F",
        #     "rules": [("F", "F+F-F-FFF+F+F-F")],
        #     "angle": 90,
        #     "expansions": 3
        # },
        # {
        #     "name": "Koch_Generalization_2",
        #     "axiom": "F+F+F+F",
        #     "rules": [("F", "F-FF+FF+F+F-F-FF+F+F-F-FF-FF+F")],
        #     "angle": 90,
        #     "expansions": 2
        # },
        # {
        #     "name": "Quadratic_Snowflake",
        #     "axiom": "F",
        #     "rules": [("F", "F-F+F+F-F")],
        #     "angle": 90,
        #     "expansions": 4
        # },
        # {
        #     "name": "Hex_Fractal",
        #     "axiom": "YF",
        #     "rules": [
        #         ("X", "YF+XF+Y"),
        #         ("Y", "XF-YF-X")
        #     ],
        #     "angle": 60,
        #     "expansions": 7
        # },
        # {
        #     "name": "Square_Curve_Variant_3",
        #     "axiom": "F+F+F+F",
        #     "rules": [("F", "F+F-F+F+F")],
        #     "angle": 90,
        #     "expansions": 3
        # },
        # {
        #     "name": "Square_Curve_Variant_4",
        #     "axiom": "F+F+F+F",
        #     "rules": [("F", "FF+F+F+F+F+F-F")],
        #     "angle": 90,
        #     "expansions": 3
        # },
        # {
        #     "name": "Bush_1",
        #     "axiom": "Y",
        #     "rules": [
        #         ("X", "X[-FFF][+FFF]FX"),
        #         ("Y", "YFX[+Y][-Y]")
        #     ],
        #     "angle": 180 / 7,
        #     "expansions": 5,
        #     "step_length": 3
        # },
        # {
        #     "name": "Bush_2",
        #     "axiom": "F",
        #     "rules": [("F", "FF+[+F-F-F]-[-F+F+F]")],
        #     "angle": 180 / 8,
        #     "expansions": 4,
        #     "step_length": 3
        # },
        {
            "name": "Bush_3",
            "axiom": "F",
            "rules": [("F", "F[+FF][-FF]F[-F][+F]F")],
            "angle": 180 / 5,
            "expansions": 4,
            "step_length": 3
        },
        # {
        #     "name": "Bush_4",
        #     "axiom": "X",
        #     "rules": [
        #         ("F", "FF"),
        #         ("X", "F[+X]F[-X]+X")
        #     ],
        #     "angle": 180 / 9,
        #     "expansions": 5,
        #     "step_length": 3
        # },
        # {
        #     "name": "Box_Fractal",
        #     "axiom": "F-F-F-F",
        #     "rules": [("F", "F-F+F+F-F")],
        #     "angle": 90,
        #     "expansions": 4
        # },
        # {
        #     "name": "Weed",
        #     "axiom": "F",
        #     "rules": [("F", "F[+F]F[-F]F")],
        #     "angle": 180 / 7,
        #     "expansions": 4,
        #     "step_length": 3
        # },
        # {
        #     "name": "Zaykov_Gallery_1",
        #     "axiom": "F",
        #     "rules": [
        #         ("F", "FXF"),
        #         ("X", "[-F+F+F]+F-F-F+")
        #     ],
        #     "angle": 60,
        #     "expansions": 4
        # },
        {
            "name": "Sierpinski_Triangle",
            "axiom": "FXF--FF--FF",
            "rules": [
                ("F", "FF"),
                ("X", "--FXF++FXF++FXF--")
            ],
            "angle": 60,
            "expansions": 6
        },
        # {
        #     "name": "Sierpinski_Carpet",
        #     "axiom": "F",
        #     "rules": [("F", "FFF[+FFF+FFF+FFF]")],
        #     "angle": 90,
        #     "expansions": 3,
        #     "step_length": 2
        # },
        # {
        #     "name": "Mosaic",
        #     "axiom": "F-F-F-F",
        #     "rules": [
        #         ("F", "F-B+FF-F-FF-FB-FF+B-FF+F+FF+FB+FFF"),
        #         ("B", "BBBBBB")
        #     ],
        #     "angle": 90,
        #     "expansions": 2,
        #     "step_length": 1
        # },
        # {
        #     "name": "Levy_C_Curve",
        #     "axiom": "F++F++F++F",
        #     "rules": [("F", "-F++F-")],
        #     "angle": 45,
        #     "expansions": 8
        # },
        # {
        #     "name": "Pentaplexity",
        #     "axiom": "F++F++F++F++F",
        #     "rules": [("F", "F++F++F+++++F-F++F")],
        #     "angle": 36,
        #     "expansions": 3
        # },
        # {
        #     "name": "Sierpinski_Carpet_Variant",
        #     "axiom": "F",
        #     "rules": [
        #         ("F", "F+F-F-F-B+F+F+F-F"),
        #         ("B", "BBB")
        #     ],
        #     "angle": 90,
        #     "expansions": 3,
        #     "step_length": 2
        # },
        # {
        #     "name": "Pentigree",
        #     "axiom": "F-F-F-F-F",
        #     "rules": [("F", "F-F++F+F-F-F")],
        #     "angle": 72,
        #     "expansions": 3
        # },
        # {
        #     "name": "Hex-7-b",
        #     "axiom": "X",
        #     "rules": [
        #         ("F", ""),
        #         ("X", "-F++F-X-F--F+Y---F--F+Y+F++F-X+++F++F-X-F++F-X+++F--F+Y--"),
        #         ("Y", "+F++F-X-F--F+Y+F--F+Y---F--F+Y---F++F-X+++F++F-X+++F--F+Y")
        #     ],
        #     "angle": 30,
        #     "expansions": 2,
        #     "step_length": 2
        # },
        # {
        #     "name": "Peano-c",
        #     "axiom": "FX",
        #     "rules": [
        #         ("F", ""),
        #         ("X", "FX-FY-FX+FY+FX+FY+FX+FY+FX-FY-FX-FY-FX-FY-FX+FY+FX"),
        #         ("Y", "FY")
        #     ],
        #     "angle": 45,
        #     "expansions": 3,
        #     "step_length": 2
        # },
        # {
        #     "name": "Border1",
        #     "axiom": "XYXYXYX+XYXYXYX+XYXYXYX+XYXYXYX",
        #     "rules": [
        #         ("F", ""),
        #         ("X", "FX+FX+FXFY-FY-"),
        #         ("Y", "+FX+FXFY-FY-FY")
        #     ],
        #     "angle": 90,
        #     "expansions": 2,
        #     "step_length": 2
        # },
        # {
        #     "name": "Doily",
        #     "axiom": "F--F--F--F--F--F",
        #     "rules": [("F", "-F[--F--F]++F--F+")],
        #     "angle": 30,
        #     "expansions": 3,
        #     "step_length": 4
        # },
        # {
        #     "name": "Maze01",
        #     "axiom": "F+F+F",
        #     "rules": [("F", "F+FF-F")],
        #     "angle": 120,
        #     "expansions": 4
        # },
        # {
        #     "name": "Maze_and_Fractal_1",
        #     "axiom": "X",
        #     "rules": [
        #         ("F", ""),
        #         ("X", "FY+FYFY-FY"),
        #         ("Y", "FX-FXFX+FX")
        #     ],
        #     "angle": 120,
        #     "expansions": 5,
        #     "step_length": 4
        # },
        {
            "name": "Moore",
            "axiom": "X",
            "rules": [
                ("X", "FX+FX+FXFYFX+FXFY-FY-FY-"),
                ("Y", "+FX+FX+FXFY-FYFXFY-FY-FY")
            ],
            "angle": 90,
            "expansions": 4,
            "step_length": 2
        },
        # {
        #     "name": "Pentant",
        #     "axiom": "X-X-X-X-X",
        #     "rules": [
        #         ("X", "FX-FX-FX+FY+FY+FX-FX"),
        #         ("Y", "FY+FY-FX-FX-FY+FY+FY")
        #     ],
        #     "angle": 72,
        #     "expansions": 2,
        #     "step_length": 3
        # },
        # {
        #     "name": "Pentl",
        #     "axiom": "F-F-F-F-F",
        #     "rules": [("F", "F-F-F++F+F-F")],
        #     "angle": 72,
        #     "expansions": 3
        # },
        # {
        #     "name": "Sierpinsk",
        #     "axiom": "L--F--L--F",
        #     "rules": [
        #         ("L", "+R-F-R+"),
        #         ("R", "-L+F+L-")
        #     ],
        #     "angle": 45,
        #     "expansions": 7
        # },
        # {
        #     "name": "Tiling1",
        #     "axiom": "X",
        #     "rules": [
        #         ("X", "F-F-F+F+FX++F-F-F+F+FX--F-F-F+F+FX"),
        #         ("F", "")
        #     ],
        #     "angle": 60,
        #     "expansions": 3,
        #     "step_length": 2
        # },
        # {
        #     "name": "ADH231a",
        #     "axiom": "F++++F",
        #     "rules": [("F", "F+F+F++++F+F+F")],
        #     "angle": 45,
        #     "expansions": 3
        # },
        # {
        #     "name": "ADH256a",
        #     "axiom": "F+F+F+F++F-F-F-F",
        #     "rules": [("F", "F+F++F+FF")],
        #     "angle": 90,
        #     "expansions": 3
        # },
        {
            "name": "ADH258a",
            "axiom": "F++F++F+++F--F--F",
            "rules": [("F", "FF++F++F++FFF")],
            "angle": 60,
            "expansions": 3
        },
        {
            "name": "MyCreation",
            "axiom": "F++F+++F++F",
            "rules": [("F", "FF[-F+F+F]F[+F-F-F]FF")],
            "angle": 60,
            "expansions": 3
        }
    ]

    for system in l_systems:
        print(f"Creating {system['name']}...")

        lsys = LSystem(
            axiom=system["axiom"],
            rules=system["rules"],
            angle=system["angle"]
        )

        step_length = system.get("step_length", 5)

        filename = f"{system['name']}.gif"
        visualize_lsystem(
            lsystem=lsys,
            expansions=system["expansions"],
            step_length=step_length,
            filename=filename
        )
