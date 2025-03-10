from BIA.GraphMethods.graph_soma import graph_soma_optimization
from BIA.ex_7_soma.soma_solver import SomaSolver
from BIA.function import all_functions


def run_soma():
    for func in all_functions():
        soma_solver = SomaSolver(func[0])
        soma_solver.search()
        graph_soma_optimization(func[0], soma_solver.populations)
        # graph_soma_optimization_complex(func[0], soma_solver.migration_history)

    pass
