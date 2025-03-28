from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from numpy import ndarray


@dataclass
class ExplorerPositionSnap:
    row: int
    col: int
    state: int
    iteration: int


class Ql:
    def __init__(self, maze_map: ndarray):
        self.maze_map: ndarray = maze_map

        height, width = maze_map.shape
        self.height: int = height
        self.width: int = width
        total_states: int = height * width

        # R matice
        self.R = np.full((total_states, total_states), -1, dtype=float)

        # mozne pohyby mysi
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        # Vytvoreni R matice z mapy (maticova reprezentace grafu)
        for i in range(height):
            for j in range(width):
                # prekazky zustavaji -1
                if maze_map[i, j] == -1:
                    continue

                current_state = i * width + j

                # 4 mozne pohyby
                for di, dj in moves:
                    ni, nj = i + di, j + dj

                    # kontrola zda je pozice v matici
                    if 0 <= ni < height and 0 <= nj < width:
                        # nenastavuje se pokud je prekazka
                        if maze_map[ni, nj] == -1:
                            continue

                        next_state = ni * width + nj
                        self.R[current_state, next_state] = maze_map[ni, nj]

        # Q matice pouze s 0
        self.Q = np.zeros((total_states, total_states), dtype=float)

    # explorace
    def explore(self, iterations: int,  # pocet iteraci
                learning_rate: float,
                initial_state: int = 0,  # pocatni pozice
                epsilon_start: float = 0.9,  # pocatecni epsilon
                epsilon_end: float = 0.1,  # konecny epsilon
                random_teleport_probability: float = 0.05):  # pravdepodobnost teleportace na nahodnou pozici kazdou iteraci

        if iterations < 1 or learning_rate <= 0 or learning_rate > 1:
            raise Exception('Invalid parameters')

        if initial_state < 0 or initial_state >= self.height * self.width:
            raise Exception('Invalid initial state')

        epsilon: float = epsilon_start
        epsilon_decrease: float = (epsilon_start - epsilon_end) / iterations

        current_state: int = initial_state

        # validni stavy
        valid_states = []
        for s in range(self.height * self.width):
            if -1 not in self.R[s, :]:
                valid_states.append(s)

        for iteration in range(iterations):
            # nahodne teleportovani na jinou pozici
            if np.random.rand() < random_teleport_probability:
                if valid_states:
                    current_state = np.random.choice(valid_states)

            # ziskani validnich pohybu v aktualnim stavu
            valid_moves = np.where(self.R[current_state] != -1)[0]

            if len(valid_moves) == 0:
                # zadne validni pohyby, probehne teleportace
                if valid_states:
                    current_state = np.random.choice(valid_states)
                continue

            # rozhodnuti, zda se bude pohybovat nahodne nebo podle Q hodnot
            # rozhoduje se podle epsilon, ktere se postupne snizuje (ke konci bude probihat spise vyuzivani Q hodnot)
            if np.random.rand() < epsilon:
                # nahodny pohyb
                next_state = np.random.choice(valid_moves)
            else:
                # pohyb podle nejlepsi Q hodnoty
                valid_q_values = [self.Q[current_state, move] for move in valid_moves]
                next_state = valid_moves[np.argmax(valid_q_values)]

            # hodnota R pro aktualni stav, nasledujici stav
            R_next_value = self.R[current_state, next_state]

            # ziskani maximalni Q hodnoty pro nasledujici stav
            next_valid_moves = np.where(self.R[next_state] != -1)[0]

            if len(next_valid_moves) > 0:
                next_q_values = [self.Q[next_state, move] for move in next_valid_moves]
                max_next_q = np.max(next_q_values)
            else:
                # pokud neni zadny validni pohyb, max_next_q je 0
                max_next_q = 0

            # aktualizace Q hodnoty
            self.Q[current_state, next_state] = R_next_value + learning_rate * max_next_q
            current_state = next_state

            epsilon = max(epsilon_end, epsilon - epsilon_decrease)

    def find(self, position: Tuple[int, int],
             max_iterations: int) -> List[ExplorerPositionSnap]:

        row, col = position
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise Exception('Invalid initial state')

        if max_iterations < 1:
            raise Exception('Invalid iterations')

        if self.maze_map[row, col] == -1:
            raise Exception('Obstacle in initial position')

        # pocatecni pozice
        current_state = row * self.width + col

        # historie pro vizualizaci
        positions_history = [
            ExplorerPositionSnap(row=row, col=col, state=current_state, iteration=0)
        ]

        # hledani pomoci Q hodnot
        for iteration in range(1, max_iterations + 1):
            # validni pohyby
            valid_moves = np.where(self.R[current_state] != -1)[0]

            if len(valid_moves) == 0:
                # zadne validni pohyby, konec
                break

            # Q hodnoty pro validni pohyby
            valid_q_values = [self.Q[current_state, action] for action in valid_moves]

            # kontrola jestli jsou vsechny Q hodnoty stejne (tedy vsechny jsou asi 0)
            if len(set(valid_q_values)) <= 1 or np.max(valid_q_values) == 0:
                # pokud ano, tak nahodne vybran pohyb
                next_state = np.random.choice(valid_moves)
            else:
                # pokud ne, je vybran pohyb s nejlepsi Q hodnotou
                next_state = valid_moves[np.argmax(valid_q_values)]

            next_row = next_state // self.width
            next_col = next_state % self.width
            positions_history.append(
                ExplorerPositionSnap(
                    row=next_state // self.width,
                    col=next_state % self.width,
                    state=next_state,
                    iteration=iteration
                )
            )

            current_state = next_state

            # pokud byla nalezena odmena, konec
            if self.maze_map[next_row, next_col] > 0:
                break

        return positions_history
