from typing import Type

import numpy as np

from function import OptimizationFunction


class BlindSearch:
    def __init__(self, optimization_function: Type[OptimizationFunction], dimension: int):
        self.function_class = optimization_function
        self.dimension = dimension
        self.lB, self.uB = optimization_function.recommended_range()
        self.all_solutions = []
        self.best_solution = None
        self.best_result = None

    def search(self, iterations: int):
        # Nejlepsi vstup do funkce
        best_solution = None
        # Nejlepsi vysledek
        best_result = float('inf')
        self.all_solutions = []

        for _ in range(iterations):

            # Vygenerovani nahodneho reseni v dane mezi
            solution = np.random.uniform(self.lB, self.uB, self.dimension)

            # Zjisteni vysledku reseni
            result = self.function_class.evaluate(solution)

            # Pokud je reseni lepsi (mensi), nahradim stavajici
            if result < best_result:
                best_solution = solution
                best_result = result
                self.all_solutions.append(solution)

        self.best_solution = best_solution
        self.best_result = best_result
