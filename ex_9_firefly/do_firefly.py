from GraphMethods.graph_firefly import graph_firefly_optimization
from ex_9_firefly.firefly_solver import FireflySolver
from function import all_functions


def run_firefly():
    for func in all_functions():
        ff = FireflySolver(func[0])
        ff.search()
        graph_firefly_optimization(ff.func_optimization, ff.swarms)

    pass
