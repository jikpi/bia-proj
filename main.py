from ex1_2_perceptron import perceptron_solver
from ex1_2_perceptron import xor_solver

if __name__ == "__main__":
    # Vystupy ve slozce 'Outputs'

    print('Perceptron ########################')
    # hadam ze nelze nejak pouzit pro 3 tridy ("nad, na, pod linkou"), kdyz jsem to zkusil tak to neslo (trida 'na' lince
    # byla spise neco jako 'zhruba kolem linky').
    perceptron_solver.test_task()

    print('XOR ########################')
    xor_solver.test_task()

pass
