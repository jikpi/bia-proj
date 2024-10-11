from typing import Tuple, Type

import numpy as np


class OptimizationFunction:

    @classmethod
    def evaluate(cls, x: np.ndarray) -> float:
        raise NotImplementedError()

    @classmethod
    def recommended_range(cls) -> tuple[float, float]:
        raise NotImplementedError()


def all_functions() -> list[tuple[Type[OptimizationFunction], str]]:
    return [
        (Sphere, "Sphere"),
        (Schwefel, "Schwefel"),
        (Rosenbrock, "Rosenbrock"),
        (Rastrigin, "Rastrigin"),
        (Griewank, "Griewank"),
        (Levy, "Levy"),
        (Michalewicz, "Michalewicz"),
        (Zakharov, "Zakharov"),
        (Ackley, "Ackley")
    ]


class Sphere(OptimizationFunction):

    @classmethod
    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(x ** 2)

    @classmethod
    def recommended_range(self) -> tuple[float, float]:
        return -100, 100


class Schwefel(OptimizationFunction):

    @classmethod
    def evaluate(cls, x: np.ndarray) -> float:
        return (418.9829 * x.size) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    @classmethod
    def recommended_range(cls) -> tuple[float, float]:
        return -500, 500


class Rosenbrock(OptimizationFunction):

    @classmethod
    def evaluate(cls, x: np.ndarray) -> float:
        # x[:-1] = x_i, x[1:] = x_(i+1)
        return np.sum((100 * (x[1:] - x[:-1] ** 2) ** 2) + (x[:-1] - 1) ** 2)

    @classmethod
    def recommended_range(cls) -> tuple[float, float]:
        return -2.048, 2.048


class Rastrigin(OptimizationFunction):

    @classmethod
    def evaluate(cls, x: np.ndarray) -> float:
        return (10 * x.size) + np.sum(x ** 2 - (10 * np.cos(2 * np.pi * x)))

    @classmethod
    def recommended_range(cls) -> tuple[float, float]:
        return -5.12, 5.12


class Griewank(OptimizationFunction):

    @classmethod
    def evaluate(cls, x: np.ndarray) -> float:
        # np.arange(1, x.size + 1) = cislo indexu i pro kazdy prvek v soucinu
        return (1 / 4000) * np.sum(x ** 2) - np.prod(np.cos(x / np.sqrt(np.arange(1, x.size + 1)))) + 1

    @classmethod
    def recommended_range(cls) -> tuple[float, float]:
        return -600, 600


class Levy(OptimizationFunction):

    @classmethod
    def evaluate(cls, x: np.ndarray) -> float:
        # w[-1] = w_d
        # w[:-1] = w_i (i=1,2,...,D-1)
        w = 1 + (x - 1) / 4
        return (np.sin(np.pi * w[0]) ** 2 +  # sin^2(pi*w_1)
                np.sum((w[:-1] - 1) ** 2 * (
                        1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2)) +  # sum((w_i-1)^2*(1+10sin^2(pi*w_i+1)))
                (w[-1] - 1) ** 2  # (w_D-1)^2
                * (1 + np.sin(2 * np.pi * w[-1]) ** 2))  # *(1+sin^2(2*pi*w_D))

    @classmethod
    def recommended_range(cls) -> tuple[float, float]:
        return -10, 10


class Michalewicz(OptimizationFunction):

    @classmethod
    def evaluate(cls, x: np.ndarray) -> float:
        m = 10
        return -np.sum(np.sin(x) * np.sin(np.arange(1, x.size + 1) * x ** 2 / np.pi) ** (2 * m))

    @classmethod
    def recommended_range(cls) -> tuple[float, float]:
        return 0, np.pi


class Zakharov(OptimizationFunction):

    @classmethod
    def evaluate(cls, x: np.ndarray) -> float:
        return np.sum(x ** 2) + np.sum(0.5 * np.arange(1, x.size + 1) * x) ** 2 + np.sum(
            0.5 * np.arange(1, x.size + 1) * x) ** 4

    @classmethod
    def recommended_range(cls):
        # return -5, 10
        return -10, 10


class Ackley(OptimizationFunction):

    @classmethod
    def evaluate(cls, x: np.ndarray) -> float:
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = x.size
        return (-a * np.exp(-b * np.sqrt((1 / d) * np.sum(x ** 2))) -
                np.exp((1 / d) * np.sum(np.cos(c * x))) + a + np.exp(1))

    @classmethod
    def recommended_range(cls) -> Tuple[float, float]:
        return -45, 45
        # return -32.768, 32.768
