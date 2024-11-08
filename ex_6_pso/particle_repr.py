from typing import Type

import numpy as np

from function import OptimizationFunction


class Particle:
    def __init__(self, params: np.ndarray, velocity: np.ndarray):
        self.params = params
        self.velocity = velocity
        self.best_params = params

    def __repr__(self):
        return f"DeSolution(vector={self.velocity})"

    def eval(self, optimization_function: Type[OptimizationFunction]) -> float:
        return optimization_function.evaluate(self.params)

    def get_result(self) -> float:
        pass

    def __eq__(self, other):
        return np.array_equal(self.velocity, other.velocity) and np.array_equal(self.params, other.params)


class Swarm:
    def __init__(self, pop_size: int, optimization_function: Type[OptimizationFunction], vel_min: float,
                 vel_max: float, dimensions: int = 2):
        self.pop_size = pop_size
        self.lb, self.ub = optimization_function.recommended_range()

        self.particles = []

        # inicializace particles
        for i in range(pop_size):
            params = np.random.uniform(self.lb, self.ub, dimensions)
            velocity = np.random.uniform(vel_min, vel_max, dimensions)
            self.particles.append(Particle(params, velocity))

        # nalezeni nejlepsi particle
        best_particle = self.particles[0]
        for particle in self.particles:
            if particle.eval(optimization_function) < best_particle.eval(optimization_function):
                best_particle = particle

        self.best_particle = best_particle
