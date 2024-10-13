import numpy as np
from typing import Type, List

from function import OptimizationFunction


# Objekt pro ulozeni vsech vygenerovanych sousedu pro vizualizaci
class IterationNeighbourData:
    def __init__(self, neighbors: List[np.ndarray], best_neighbor: np.ndarray):
        self.neighbors = neighbors
        self.best_neighbor = best_neighbor


class HillClimbing:
    def __init__(self, optimization_function: Type[OptimizationFunction], dimension: int):
        self.function_class = optimization_function
        self.dimension = dimension
        self.lB, self.uB = optimization_function.recommended_range()
        self.iteration_data = []
        self.best_solution = None
        self.best_result = None
        self.range_scale_divisor = 10  # Skalovani (podle rozsahu funkce) pro rozeseti neighbors (Zvetsit pro mensi rozsah)
        self.max_unsuccessful_fraction = 10  # Delitel maximalniho poctu iteraci pro neuspesne pokusy (Zmensit pro vice pokusu)

    # Vypocet skalovani
    def range_scale(self):
        return (self.uB - self.lB) / self.range_scale_divisor

    # Vygenerovani sousedu pro dane reseni
    def generate_neighbors(self, current_solution: np.ndarray, num_neighbors: int = 50) -> List[np.ndarray]:
        neighbors = []
        for _ in range(num_neighbors):
            # Okruh rozeseti neighbors
            neighbor = current_solution + np.random.normal(0, self.range_scale(), self.dimension)
            neighbor = np.clip(neighbor, self.lB, self.uB)
            neighbors.append(neighbor)
        return neighbors

    def search(self, max_iterations: int):
        current_solution = np.random.uniform(self.lB, self.uB, self.dimension)  # Vygenerovani nahodne prvni solution
        current_result = self.function_class.evaluate(current_solution)  # Evaluace prvni solution
        max_unsuccessful = max_iterations // self.max_unsuccessful_fraction

        self.best_solution = current_solution
        self.best_result = current_result

        for _ in range(max_iterations):
            neighbors = self.generate_neighbors(current_solution)  # Vygenerovani sousedu kolem bodu
            best_neighbor = None
            best_neighbor_result = float('inf')

            # Prochazeni vygenerovanych sousedu a vyber nejlepsiho
            for neighbor in neighbors:
                result = self.function_class.evaluate(neighbor)
                if result < best_neighbor_result:
                    best_neighbor = neighbor
                    best_neighbor_result = result

            # Pokud je nejlepsi soused lepsi nez aktualni reseni, nahradim aktualni reseni
            self.iteration_data.append(IterationNeighbourData(neighbors, self.best_solution))
            if best_neighbor_result < current_result:
                current_solution = best_neighbor
                current_result = best_neighbor_result

                if current_result < self.best_result:
                    self.best_solution = current_solution
                    self.best_result = current_result
            else:
                # Vlastni mirna uprava algoritmu
                if max_unsuccessful <= 0:
                    break

                # Pokud se nedari najit nic lepsiho, zmensuje se okruh rozeseti sousedu
                self.range_scale_divisor += 1
                max_unsuccessful -= 1
            # break
