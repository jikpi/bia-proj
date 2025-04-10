import numpy as np
from typing import Optional


class Ifs:
    def __init__(self, model: np.ndarray,  # model
                 seed: Optional[int] = None):
        self.model = model
        self.rng = np.random.RandomState(seed)
        self.current_point = np.zeros(3)

        # stejna pravdepodobnost pro kazdou transformaci
        num_transformations = model.shape[0]
        self.probabilities = np.ones(num_transformations) / num_transformations

        # historie bodu
        self.point_history = np.zeros((3, 1))

    # transformace bodu
    def transform(self, transform_index: int) -> np.ndarray:
        transformation = self.model[transform_index]
        a, b, c, d, e, f, g, h, i, j, k, l = transformation
        x, y, z = self.current_point

        x_new = a * x + b * y + c * z + j
        y_new = d * x + e * y + f * z + k
        z_new = g * x + h * y + i * z + l

        return np.array([x_new, y_new, z_new])

    # provedeni X transformaci a ulozeni vsech pozic
    def step(self, count: int) -> None:
        new_points = np.zeros((3, count))

        for i in range(count):
            transformation_index = self.rng.choice(len(self.model), p=self.probabilities)
            self.current_point = self.transform(transformation_index)
            new_points[:, i] = self.current_point

        self.point_history = np.hstack([self.point_history, new_points])
