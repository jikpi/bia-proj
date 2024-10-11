from typing import Type

from ex_1_BlindSearch.blind_search import BlindSearch
from function import *
from graph import graph_3d_generic


def do_all_blind_search(dimensions: int = 2, iterations: int = 100000):
    for func, name in all_functions():
        print(f'Blindsearch {name}')
        current_blind_search = BlindSearch(func, dimensions)
        current_blind_search.search(iterations)
        graph_3d_generic(func, solutions=current_blind_search.all_solutions,
                         save_result=True, save_gif=True,
                         input_text=f'{name} Blind search')
