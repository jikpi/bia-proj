from BIA.GraphMethods.graph_pso import graph_pso_optimization
from BIA.ex_6_pso.pso_solver import PsoSolver
from BIA.function import all_functions


def run_pso():
    for func in all_functions():
        pso = PsoSolver(func[0])
        pso.search()
        graph_pso_optimization(func[0], pso.swarms)

    pass
