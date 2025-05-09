from ex1_2_perceptron import perceptron_solver
from ex1_2_perceptron import xor_solver
from ex3_hopfield.hopfield_solver import hopfield_solve
from ex4_qlearning.ql_solver import maze_solve
from ex6_lsystems.ls_solver import ls_solve
from ex7_ifs.ifs_solver import ifs_solve
from ex8_tea.tea_set_solver import ts_solve
from ex9_terrain.terrain_solver import terrain_solve
from ex10_lm.lm_solver import lm_solve, lm_solve_ann

if __name__ == "__main__":
    # Vystupy ve slozce 'Outputs'

    # print('Perceptron ########################')
    # perceptron_solver.test_task()
    #
    # print('XOR ########################')
    # xor_solver.test_task()

    # print('HOPFIELD ########################')
    # hopfield_solve()
    #
    # print('Maze ########################')
    # maze_solve()
    # pass
    # print('L Systems ########################')
    # ls_solve()

    # print('IFS ########################')
    # ifs_solve()

    #print('TEA ########################')
    #ts_solve()

    #print('Terrain ########################')
    #terrain_solve()

    # todo: pytorch pro CPU: 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu'
    # todo: + 'pip install scikit-learn'
    # pro jistotu jsem dal outputy do slozky 'ex10_lm' pro tento task

    print('LM ########################')
    lm_solve()
    lm_solve_ann()

pass
