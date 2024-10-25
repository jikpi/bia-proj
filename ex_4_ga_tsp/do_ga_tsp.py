from GraphMethods.graph_ga_tsp import create_city_path_animation, visualize_city_evolution
from ex_4_ga_tsp.ga_tsp_solver import GenAlgTspSolver


def run_ga_tsp():
    ga_stp = GenAlgTspSolver(g=200, initial_city_seed=100, solution_seed=200, np=20)
    ga_stp.search()
    visualize_city_evolution(ga_stp.ent_bygen_list, 1000, 1000, city_seed=ga_stp.initial_city_seed,
                             solution_seed=ga_stp.solution_seed, city_size=ga_stp.np)
    pass
