import numpy as np
from typing import Type, List
from function import OptimizationFunction


class IterationNeighbourData:
    def __init__(self, neighbors: List[np.ndarray], best_neighbor: np.ndarray, accepted_neighbor: np.ndarray,
                 temperature: float):
        self.neighbors = neighbors
        self.best_neighbor = best_neighbor
        self.accepted_neighbor = accepted_neighbor
        self.temperature = temperature


class SimulatedAnnealing:
    def __init__(self, optimization_function: Type[OptimizationFunction], dimension: int):
        self.function_class = optimization_function
        self.dimension = dimension
        self.lB, self.uB = optimization_function.recommended_range()
        self.iteration_data = []
        self.best_solution = None  # nejlepsi reseni behem celeho hledani
        self.best_result = None
        self.range_scale_divisor = 10
        self.initial_temperature = 400
        self.cooling_rate = 0.955

    def range_scale(self):
        return (self.uB - self.lB) / self.range_scale_divisor

    def generate_neighbors(self, current_solution: np.ndarray, num_neighbors: int = 50) -> List[np.ndarray]:
        neighbors = []
        for _ in range(num_neighbors):
            neighbor = current_solution + np.random.normal(0, self.range_scale(), self.dimension)
            neighbor = np.clip(neighbor, self.lB, self.uB)
            neighbors.append(neighbor)
        return neighbors

    def acceptance_probability(self, current_result: float, new_result: float, temperature: float) -> float:
        # pokud je novy vysledek lepsi, vrati se vzdy 1
        if new_result < current_result:
            return 1.0
        # pokud je horsi, vraci se pravdepodobnost podle vzorce
        return np.exp((current_result - new_result) / temperature)  # e^(-(f(x_1)-f(x))/T)

    def search(self, max_iterations: int):
        # momentalni reseni, nemusi byt nejlepsi
        current_solution = np.random.uniform(self.lB, self.uB, self.dimension)
        current_result = self.function_class.evaluate(current_solution)
        temperature = self.initial_temperature

        self.best_solution = current_solution
        self.best_result = current_result

        for _ in range(max_iterations):
            # vygenerovani sousedu
            neighbors = self.generate_neighbors(current_solution)

            # nejlepsi soused v dane iteraci (ignoruje annealing, pro vizualizaci)
            best_neighbor = None
            best_neighbor_result = float('inf')
            # prijaty soused v dane iteraci (podle pravdepodobnosti)
            accepted_neighbor = None

            for neighbor in neighbors:
                result = self.function_class.evaluate(neighbor)
                if result < best_neighbor_result:
                    best_neighbor = neighbor
                    best_neighbor_result = result

                # vypocet pravdepodobnosti z funkce, porovnava se oproti np.random.random() ktere je mezi 0 a 1
                if self.acceptance_probability(current_result, result, temperature) > np.random.random():
                    # pokud je pravdepodobnost vetsi nez nahodne cislo, prijme se novy soused
                    # jakozto reseni, kolem ktereho se bude hledat dale
                    accepted_neighbor = neighbor
                    current_solution = neighbor
                    current_result = result

                    if current_result < self.best_result:
                        self.best_solution = current_solution
                        self.best_result = current_result

            self.iteration_data.append(IterationNeighbourData(neighbors, best_neighbor, accepted_neighbor, temperature))

            temperature *= self.cooling_rate
            print(f'teplota: {temperature}')
            # pokud je teplota (skoro) nula nebo mensi, break
            if temperature < 1e-8:
                break

        return self.best_solution, self.best_result
