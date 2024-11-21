from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class PathPoint:
    position: np.ndarray
    fitness: float
    step: float
    leader_index: int


@dataclass
class IndividualTrajectory:
    start_position: np.ndarray
    start_fitness: float
    paths: Dict[int, List[PathPoint]]
    final_position: np.ndarray
    final_fitness: float


class MigrationSnapshot:

    def __init__(self, population_size: int):
        self.migration_number: int = 0
        self.population_size: int = population_size
        self.trajectories: List[IndividualTrajectory] = [None] * population_size
        self.population_diversity: float = 0.0
        self.best_fitness: float = float('inf')
        self.best_position: np.ndarray | None = None

    def add_trajectory(self,
                       individual_index: int,
                       start_position: np.ndarray,
                       start_fitness: float,
                       final_position: np.ndarray,
                       final_fitness: float):
        self.trajectories[individual_index] = IndividualTrajectory(
            start_position=start_position.copy(),
            start_fitness=start_fitness,
            paths={},
            final_position=final_position.copy(),
            final_fitness=final_fitness
        )

        if final_fitness < self.best_fitness:
            self.best_fitness = final_fitness
            self.best_position = final_position.copy()

    def add_path_point(self,
                       individual_index: int,
                       leader_index: int,
                       position: np.ndarray,
                       fitness: float,
                       step: float):
        if self.trajectories[individual_index] is None:
            raise ValueError(f"Trajectory for individual {individual_index} not initialized")

        if leader_index not in self.trajectories[individual_index].paths:
            self.trajectories[individual_index].paths[leader_index] = []

        self.trajectories[individual_index].paths[leader_index].append(
            PathPoint(
                position=position.copy(),
                fitness=fitness,
                step=step,
                leader_index=leader_index
            )
        )
