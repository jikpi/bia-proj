from BIA.GraphMethods.graph_ga_de import graph_de_optimization
from BIA.ex_5_ga_de.ga_de_solver import DiffEvolSolver
from BIA.function import all_functions


def run_ga_de():
    init_seed = 10101
    solution_seed = 2020
    npop = 10
    g = 30

    for func in all_functions():
        de = DiffEvolSolver(func[0], vec_init_seed=init_seed, solution_seed=solution_seed, npop=npop, g=g)
        de.search()
        graph_de_optimization(func[0], de.gens, init_seed=init_seed, solution_seed=solution_seed)

    pass
