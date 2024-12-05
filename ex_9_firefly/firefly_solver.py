from copy import deepcopy
from random import random
from typing import Type, List

import numpy as np

from ex_9_firefly.firefly_repr import Swarm
from function import OptimizationFunction


class FireflySolver:
    def __init__(self, optimization_function: Type[OptimizationFunction],
                 pop_size: int = 20,
                 beta_0: float = 1.0,
                 alpha: float = 0.2,
                 iterations: int = 20,
                 gamma: float = 1,  # 0,1 - 10
                 dimensions: int = 3):
        self.pop_size = pop_size
        self.beta_0 = beta_0
        self.func_optimization: Type[OptimizationFunction] = optimization_function
        self.swarms: List[Swarm] = []
        self.dimensions = dimensions
        self.alpha = alpha
        self.iterations = iterations
        self.ub = optimization_function.recommended_range()[1]
        self.lb = optimization_function.recommended_range()[0]
        self.gamma = gamma

        init_swarm = Swarm(pop_size=pop_size, optimization_function=optimization_function, dimensions=dimensions)
        self.swarms.append(init_swarm)

        pass

    def search(self):
        iteration = 0
        while iteration < self.iterations:
            # epsilon = np.random.normal(0, 1, self.pop_size)

            # alpha = self.alpha * (0.95) ** iteration  # scale vzhledem k iteraci

            for i, moving_ffly in enumerate(self.swarms[-1].particles):
                for j, observed_ffly in enumerate(self.swarms[-1].particles):
                    if i == j:
                        continue

                    r = moving_ffly.euclidean_distance(observed_ffly)
                    moving_intensity = self.func_optimization.evaluate(moving_ffly.params) * np.exp(-self.gamma * r)
                    observed_intensity = self.func_optimization.evaluate(observed_ffly.params) * np.exp(-self.gamma * r)
                    if observed_intensity < moving_intensity:
                        # hybe se

                        # beta = self.beta_0 / (1 + (r ** 2))
                        beta = self.beta_0 * np.exp(-self.gamma * r ** 2)

                        new_params = (moving_ffly.params + beta * (
                                observed_ffly.params - moving_ffly.params) +
                                      (self.alpha * np.random.normal(size=self.dimensions) * (  # epsilon
                                              self.swarms[-1].ub - self.swarms[-1].lb)))  # meritko hledani
                        new_params = np.clip(new_params, self.lb, self.ub)
                        moving_ffly.params = new_params

            best_eval = self.swarms[-1].best_particle.eval(self.func_optimization)
            for particle in self.swarms[-1].particles:
                if particle.eval(self.func_optimization) < best_eval:
                    best_eval = particle.eval(self.func_optimization)
                    self.swarms[-1].best_particle = particle

            iteration += 1
            self.swarms.append(deepcopy(self.swarms[-1]))
