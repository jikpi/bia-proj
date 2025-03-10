from copy import deepcopy
from typing import Type

import numpy as np

from BIA.ex_7_soma.soma_repr import Population
from BIA.function import OptimizationFunction


class SomaSolver:
    def __init__(self, optimization_function: Type[OptimizationFunction],
                 pop_size: int = 10,  # velikost populace
                 migration_max: int = 10,  # max pocet migraci
                 path_length: float = 1.1,  # max delka cesty
                 step_size: float = 0.11,  # velikost kroku
                 prt: float = 0.5,  # perturbation vektor
                 min_div: float | None = 1e-5,  # minimalni diversita
                 dimensions: int = 2):
        self.pop_size = pop_size
        self.migration_max = migration_max
        self.path_length = path_length
        self.step_size = step_size
        self.prt = prt
        self.min_div: float | None = min_div
        self.func_optimization = optimization_function
        self.populations = []
        self.dimensions = dimensions

        init_population = Population(pop_size, optimization_function, dimensions)
        self.populations.append(init_population)

        # self.migration_history: List[MigrationSnapshot] = []

    def calculate_diversity(self, population: 'Population') -> float:
        # vypocet maximalni vzdalenosti mezi jakokolivmi 2 individuals
        max_distance = 0
        for i in range(self.pop_size):
            for j in range(i + 1, self.pop_size):
                dist = np.linalg.norm(population.individuals[i].params - population.individuals[j].params)
                max_distance = max(max_distance, dist)
        return max_distance

    def search(self):
        migration = 0
        while migration < self.migration_max:
            curr_pop: Population = deepcopy(self.populations[-1])
            population_diversity = self.calculate_diversity(curr_pop)

            if self.min_div is not None and population_diversity < self.min_div:
                print(
                    f'Stopping *{self.func_optimization.__name__}* since diversity is smaller than min div. ({population_diversity} < {self.min_div})')
                break

            # leader
            best_result = float('inf')
            leader_index = 0
            for i, individual in enumerate(curr_pop.individuals):
                result = self.func_optimization.evaluate(individual.params)
                if result < best_result:
                    best_result = result
                    leader_index = i

            curr_pop.leader = leader_index

            leader = curr_pop.individuals[leader_index]

            # vsichni ostani
            for i, individual in enumerate(curr_pop.individuals):
                if i != leader_index:
                    start_params = individual.params.copy()
                    start_result = self.func_optimization.evaluate(start_params)
                    best_params = start_params.copy()
                    best_result = start_result

                    prt_vector = np.random.rand(self.dimensions)  # vektor mutace
                    for j in range(len(prt_vector)):
                        if prt_vector[j] < self.prt:
                            prt_vector[j] = 1
                        else:
                            prt_vector[j] = 0

                    # kroky po ceste vytvorene prt vektorem a leaderem
                    current_step_location = 0
                    while current_step_location <= self.path_length:
                        new_params = individual.params + (
                                leader.params - individual.params) * current_step_location * prt_vector
                        new_params = np.clip(new_params,
                                             self.func_optimization.recommended_range()[0],
                                             self.func_optimization.recommended_range()[1])

                        current_result = self.func_optimization.evaluate(new_params)

                        if current_result < best_result:
                            best_result = current_result
                            best_params = new_params.copy()

                        current_step_location += self.step_size

                    individual.params = best_params
                    if best_result < self.func_optimization.evaluate(individual.best_params):
                        individual.best_params = best_params.copy()

            migration += 1
            self.populations.append(curr_pop)
