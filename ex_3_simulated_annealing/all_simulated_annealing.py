from GraphMethods.graph_simulated_annealing import graph_simulated_annealing
from ex_3_simulated_annealing.simulated_annealing import SimulatedAnnealing
from function import all_functions


def do_all_simulated_annealing(dimensions: int = 2, iterations: int = 100):
    for func, name in all_functions():
        print(f'Simulated Annealing {name}')
        sa = SimulatedAnnealing(func, dimensions)
        sa.search(iterations)
        graph_simulated_annealing(func, sa.iteration_data, sa.best_solution)
