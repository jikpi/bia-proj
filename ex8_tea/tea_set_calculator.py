import numpy as np
from matplotlib import cm


class SetCalculator:
    def __init__(self, fractal_type="mandelbrot",  # "julia"
                 x_size=1280, y_size=720,
                 max_iter=300,  # maximalni pocet iteraci pro zjisteni zda se bod nachazi v setu
                 mandelbrot_point=None,  # mandelbrot
                 julia_c_value=None  # julia
                 ):

        self.fractal_type = fractal_type.lower()
        self.x_size = x_size
        self.y_size = y_size
        self.max_iter = max_iter
        self.init_max_iter = max_iter

        # mandelbrot
        if self.fractal_type == "mandelbrot":
            self.x_min, self.x_max = -2.0, 1.0
            self.y_min, self.y_max = -1.0, 1.0
            self.mandelbrot_point = mandelbrot_point
            if mandelbrot_point is None:
                raise ValueError("mandelbrot point is not set")
            self.c = None
            self.z0 = 0
        # julia
        else:
            self.x_min, self.x_max = -1.5, 1.5
            self.y_min, self.y_max = -1.5, 1.5
            self.c = julia_c_value
            if self.c is None:
                raise ValueError("julia c_value not set")

            self.z0 = None

        self.current_zoom = 0
        self.max_zoom_steps = 1000
        self.zoom_amount = 0.8

    # vypocet hodnoty pro kazdy pixel
    def calculate_fractal(self):
        real_axis = np.linspace(self.x_min, self.x_max, self.x_size)
        imag_axis = np.linspace(self.y_min, self.y_max, self.y_size)

        real_values, i_values = np.meshgrid(real_axis, imag_axis)

        if self.fractal_type == "mandelbrot":
            # pro mandelbrot se meni c
            c_r = real_values
            c_i = i_values
            c = c_r + 1j * c_i

            z = np.full(c.shape, complex(self.z0, 0))
            mask = np.full(c.shape, True, dtype=bool)
            iterations = np.zeros(c.shape, dtype=int)

            # iterace
            for i in range(self.max_iter):
                z[mask] = z[mask] ** 2 + c[mask]
                # aktualizace masky na zaklade divergence kdyz |z| > 2
                mask_new = (np.abs(z) <= 2.0)
                iterations[mask & (~mask_new)] = i
                mask = mask_new

                if not np.any(mask):
                    break

            result = iterations

        else:  # julia
            # pro julia se meni z_0
            z_r = real_values
            z_i = i_values
            z = z_r + 1j * z_i

            c = np.full(z.shape, self.c)
            mask = np.full(z.shape, True, dtype=bool)
            iterations = np.zeros(z.shape, dtype=int)

            # iterace
            for i in range(self.max_iter):
                z[mask] = z[mask] ** 2 + c[mask]
                # aktualizace masky podle divergence kdyz |z| > 2
                mask_new = (np.abs(z) <= 2.0)
                iterations[mask & (~mask_new)] = i
                mask = mask_new

                if not np.any(mask):
                    break

            result = iterations

        return result

    # zoom
    def step(self):
        if self.current_zoom >= self.max_zoom_steps:
            self.current_zoom = 0

            # resetovani hranic
            if self.fractal_type == "mandelbrot":
                self.x_min, self.x_max = -2.0, 1.0
                self.y_min, self.y_max = -1.0, 1.0
            else:  # julia
                self.x_min, self.x_max = -1.5, 1.5
                self.y_min, self.y_max = -1.5, 1.5

        # vyber ciloveho bodu pro priblizeni
        if self.fractal_type == "mandelbrot":
            target_x, target_y = self.mandelbrot_point
        else:  # pro julia se priblizuje na stred
            x_center = (self.x_min + self.x_max) / 2
            y_center = (self.y_min + self.y_max) / 2
            target_x, target_y = x_center, y_center

        # vypocet novych hranic pro priblizeni
        x_center, y_center = target_x, target_y

        x_range = self.x_max - self.x_min
        y_range = self.y_max - self.y_min

        # priblizeni
        new_x_range = x_range * self.zoom_amount
        new_y_range = y_range * self.zoom_amount

        self.x_min = x_center - new_x_range / 2
        self.x_max = x_center + new_x_range / 2
        self.y_min = y_center - new_y_range / 2
        self.y_max = y_center + new_y_range / 2

        # zvyseni iteraci pro lepsi detail pri priblizeni
        self.max_iter = self.init_max_iter + self.current_zoom * 10

        self.current_zoom += 1

        return self.calculate_fractal()

    # prevod fractal hodnot na obrazek
    def render_frame(self, frame):
        # vytvoreni kopie pro zachovani originalu
        colored = np.zeros((frame.shape[0], frame.shape[1], 4))

        # vytvoreni masky pro body uvnitr mnoziny kde hodnota je 0
        inside_mask = (frame == 0)
        outside_mask = ~inside_mask

        # normalizace a obarveni bodu pouze mimo mnozinu (body uvnitr mnoziny zustavaji cerne)
        if np.any(outside_mask):
            outside_points = frame[outside_mask]
            normalized = outside_points / np.max(outside_points) if np.max(outside_points) > 0 else outside_points
            colored_outside = cm.hsv(normalized)
            # colored_outside = cm.viridis(normalized)
            # colored_outside = cm.plasma(normalized)
            # colored_outside = cm.inferno(normalized)
            # colored_outside = cm.magma(normalized)
            # colored_outside = cm.turbo(normalized)
            # colored_outside = cm.jet(normalized)

            # obarveni bodu mimo mnozinu
            colored[outside_mask] = colored_outside

        return (colored[:, :, :3] * 255).astype(np.uint8)
