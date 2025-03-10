from BIA.GraphMethods.graph_aco_tsp import visualize_ant_path, visualize_ant_evolution
from BIA.ex_8_aco.aco_tsp_solver import AcoTspSolver


def run_aco_tsp():
    aco_tsp = AcoTspSolver(city_count=40, iteration_count=50, city_seed=100,
                           alpha=1, beta=2, ant_count_ratio=1, evaporation_rate=0.5)
    aco_tsp.search()
    visualize_ant_path(aco_tsp.best_ant, aco_tsp.cities, city_seed=100)
    visualize_ant_evolution(aco_tsp.best_ants, aco_tsp.cities, city_seed=aco_tsp.city_seed, x_size=aco_tsp.x_size,
                            y_size=aco_tsp.y_size)
    pass
