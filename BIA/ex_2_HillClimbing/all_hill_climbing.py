from BIA.GraphMethods.graph_HillClimbing import graph_hill_climbing
from BIA.ex_2_HillClimbing.hill_climbing import HillClimbing
from BIA.function import *


def do_all_hill_climbing(dimensions: int = 2, iterations: int = 100):
    for func, name in all_functions():
        print(f'Hill Climbing {name}')
        hc = HillClimbing(func, dimensions)
        hc.search(iterations)
        graph_hill_climbing(func, hc.iteration_data, hc.best_solution)
