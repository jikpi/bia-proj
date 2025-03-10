from typing import Type

import numpy as np

from BIA.function import OptimizationFunction


class Individual:
    def __init__(self, params: np.ndarray):
        self.params = params
        self.best_params = params.copy()

    def __repr__(self):
        return f"Individual(params={self.params})"

    def eval(self, optimization_function: Type[OptimizationFunction]) -> float:
        return optimization_function.evaluate(self.params)

    def __eq__(self, other):
        return np.array_equal(self.params, other.params)


class Population:
    def __init__(self, pop_size: int, optimization_function: Type[OptimizationFunction], dimensions: int = 2):
        self.pop_size = pop_size
        self.lb, self.ub = optimization_function.recommended_range()

        self.individuals = [
            Individual(np.random.uniform(self.lb, self.ub, dimensions))
            for _ in range(pop_size)
        ]

        self.leader: int | None = None
