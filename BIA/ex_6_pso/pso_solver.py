from copy import deepcopy
from typing import Type

import numpy as np
from numpy.random import random

from BIA.ex_6_pso.particle_repr import Swarm
from BIA.function import OptimizationFunction


class PsoSolver:
    def __init__(self, optimization_function: Type[OptimizationFunction],
                 pop_size: int = 15,
                 migration_max=50,
                 c1: float = 2.0,
                 c2: float = 2.0,
                 vel_perc: int = 20,
                 dimensions: int = 2):
        self.pop_size = pop_size
        self.migration_max = migration_max
        self.c1 = c1
        self.c2 = c2
        self.vel_perc = vel_perc
        self.func_optimization = optimization_function
        self.swarms = []
        self.dimensions = dimensions

        # velocity je nejake procento z hledaneho prostoru, vel_min je symetricke
        self.vel_max = (self.func_optimization.recommended_range()[1] - self.func_optimization.recommended_range()[
            0]) * self.vel_perc / 100
        self.vel_min = -self.vel_max

        init_swarm = Swarm(pop_size, optimization_function, self.vel_min, self.vel_max, dimensions)
        self.swarms.append(init_swarm)

        pass

    def search(self):
        m = 0
        while m < self.migration_max:
            for i, particle in enumerate(self.swarms[-1].particles):
                # vypocet nove velocity
                w = 0.9 - ((0.5 * m) / self.migration_max)  # intertia, zmensuje se linearni s poctem mutaci
                r1 = random()
                new_velocity = particle.velocity * w + self.c1 * r1 * (
                        particle.best_params - particle.params) + r1 * self.c2 * (
                                       self.swarms[-1].best_particle.params - particle.params)

                # kontrola ohraniceni
                new_velocity = np.clip(new_velocity, self.vel_min, self.vel_max)

                # vypocet nove pozice
                new_position = particle.params + new_velocity
                new_position = np.clip(new_position, self.func_optimization.recommended_range()[0],
                                       self.func_optimization.recommended_range()[1])

                # update pozice a velocity pro particle
                particle.velocity = new_velocity
                particle.params = new_position

                # evaluace fitness
                current_fitness = self.func_optimization.evaluate(particle.params)
                personal_best_fitness = self.func_optimization.evaluate(particle.best_params)
                global_best_fitness = self.swarms[-1].best_particle.eval(self.func_optimization)

                # pokud je aktualni fitness lepsi nez personalni, tak aktualizuj personalni
                if current_fitness < personal_best_fitness:
                    particle.best_params = particle.params.copy()

                    # pokud je aktualni fitness lepsi nez globalni, tak aktualizuj globalni
                    if current_fitness < global_best_fitness:
                        self.swarms[-1].best_particle = deepcopy(particle)

            m += 1
            self.swarms.append(deepcopy(self.swarms[-1]))
