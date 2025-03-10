import numpy as np


class DeSolution:
    def __init__(self, vector: np.ndarray):
        self.vector = vector
        self.result = None

    def __repr__(self):
        return f"DeSolution(vector={self.vector})"

    def solve(self, optimization_function):
        if self.result is None:
            self.result = optimization_function.evaluate(self.vector)

        return self.result

    def get_result(self) -> float:
        if self.result is None:
            raise ValueError("Solution not evaluated")
        return self.result

    def __eq__(self, other):
        return np.array_equal(self.vector, other.vector)


# reprezentuje jednu generaci
class DeGenerationData:
    def __init__(self, generation_num: int, entities: list[DeSolution]):
        self.generation: int = generation_num
        self.entities: list[DeSolution] = entities
        self.best = None
        self.trimmed = False

    def trim_best(self, optimization_function):
        best_entity_index = min(
            range(len(self.entities)),
            key=lambda i: optimization_function.evaluate(self.entities[i].vector)
        )
        self.entities = [self.entities[best_entity_index]]
        self.trimmed = True

    def get_best(self) -> DeSolution:
        if self.trimmed:
            return self.entities[0]
        if self.best is None:
            self.best = min(self.entities, key=lambda x: x.get_result())
        return self.best

    def __repr__(self):
        return f"DeGenerationData(generation={self.generation})"

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.entities[key]
        elif isinstance(key, slice):
            return self.entities[key]
        raise TypeError(f"Invalid key type: {type(key)}")

    def __len__(self):
        return len(self.entities)
