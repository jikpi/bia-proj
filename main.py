from ex1_2_perceptron import perceptron_solver
from ex1_2_perceptron import xor_solver
from ex3_hopfield.hopfield_solver import hopfield_solve
from ex4_qlearning.ql_solver import maze_solve
from ex6_lsystems.ls_solver import ls_solve
from ex7_ifs.ifs_solver import ifs_solve

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
    print('L Systems ########################')
    ls_solve()

    print('IFS ########################')
    ifs_solve()

pass
