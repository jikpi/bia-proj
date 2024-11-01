import random
import sys
from typing import Type

import numpy as np

from ex_5_ga_de.solution_repr import DeSolution, DeGenerationData
from function import OptimizationFunction


class DiffEvolSolver:
    def __init__(self, optimization_function: Type[OptimizationFunction],
                 npop: int = 10,  # individuals
                 f: float = 0.5,  # mutation constant
                 cr: float = 0.5,  # crossover range
                 g: int = 20,  # generations
                 vec_init_seed: int = None,
                 solution_seed: int = None,
                 search_dimension: int = 2):

        if npop < 4:
            raise ValueError("npop < 4")
        if f <= 0 or f > 2:
            raise ValueError("F not in [0, 2]")
        if cr < 0 or cr > 1:
            raise ValueError("CR not in [0, 1]")

        self.npop = npop
        self.f = f
        self.cr = cr
        self.g = g
        self.vec_init_seed = vec_init_seed
        self.solution_seed = solution_seed
        self.search_dimension = search_dimension
        self.func_optimization = optimization_function
        self.gens: list[DeGenerationData] = []
        self.lb = optimization_function.recommended_range()[0]
        self.ub = optimization_function.recommended_range()[1]
        self.finished = False

        # master rng pro deterministicke generovani seedu do dalsich RNG
        self.master_rng = random.Random(solution_seed)

        # prvotni populace
        initial_pop_vector: list[DeSolution] = []
        init_rng = random.Random(vec_init_seed)
        for i in range(npop):
            vector = np.zeros(search_dimension, dtype=float)
            for j in range(search_dimension):
                vector[j] = init_rng.uniform(self.lb, self.ub)
            initial_pop_vector.append(DeSolution(vector))

        self.gens.append(DeGenerationData(0, initial_pop_vector))

        pass

    def indices_3_unique(self, current_index: int) -> tuple[int, int, int]:
        rng_ind = random.Random(self.master_rng.randrange(sys.maxsize))
        all_indices = [i for i in range(self.npop)]
        all_indices.remove(current_index)
        r1, r2, r3 = rng_ind.sample(all_indices, 3)
        return r1, r2, r3

    def search(self):
        if self.finished:
            return

        for gen in range(self.g):
            # posledni generace
            current_gen = self.gens[-1].entities[:]
            # nova generace
            new_gen = []

            rng_crossover = random.Random(self.master_rng.randrange(sys.maxsize))
            for i, solution in enumerate(current_gen):
                # vyber 3 unikatnich indexu
                r1, r2, r3 = self.indices_3_unique(i)
                # vektor mutace
                v = (current_gen[r1].vector - current_gen[r2].vector) * self.f + current_gen[r3].vector
                # clip
                v = np.clip(v, self.lb, self.ub)
                # potomek
                u = np.zeros(self.search_dimension, dtype=float)

                j_rand = rng_crossover.randint(0, self.search_dimension - 1)
                for j in range(self.search_dimension):
                    if rng_crossover.uniform(0, 1) < self.cr or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = solution.vector[j]

                new_solution = DeSolution(u)
                f_u = new_solution.solve(self.func_optimization)
                f_current = solution.solve(self.func_optimization)

                # pokud je potomek lepsi nebo stejne dobry jako rodic, tak nahrazuje rodice
                if f_u <= f_current:
                    new_gen.append(new_solution)
                else:
                    new_gen.append(solution)

            self.gens.append(DeGenerationData(gen + 1, new_gen))

            # if len(self.gens) > 1:
            #     self.gens[-2].trim_best(self.func_optimization)

        self.finished = True
