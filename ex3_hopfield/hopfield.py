import numpy as np


class HopfieldNetwork:

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.size = height * width
        self.weights = np.zeros((self.size, self.size))

    def learn_pattern(self, pattern):
        if len(pattern) != self.size:
            raise ValueError()

        # prevod 0 na -1
        pattern_bipolar = np.where(pattern == 0, -1, 1)

        # M
        M = pattern_bipolar.reshape(-1, 1)

        # M transponovane
        Mt = pattern_bipolar.reshape(1, -1)

        # W = M * M^T
        W = np.matmul(M, Mt)

        # W = W - I
        identity = np.identity(self.size)
        W = W - identity

        # P = W + P
        self.weights += W

    def recover_pattern(self, pattern, max_iterations=50):
        # prevod 0 na -1
        current_pattern = np.where(pattern == 0, -1, 1)

        # asynchronni pattern recovery
        recovery_iterations = 0
        for iteration in range(max_iterations):
            last_pattern = np.copy(current_pattern)

            for i in range(self.size):
                # vypocet aktivace pro i-ty neuron
                activation = np.dot(self.weights[i], current_pattern)

                # signum (theta=0)
                if activation > 0:
                    current_pattern[i] = 1
                elif activation < 0:
                    current_pattern[i] = -1

            # kontrola
            if np.array_equal(current_pattern, last_pattern):
                recovery_iterations += 1

            if recovery_iterations >= 5:
                break

        result = np.where(current_pattern == -1, 0, 1)

        return result

    def print_pattern(self, pattern, shape):
        reshaped = pattern.reshape(shape)
        for row in reshaped:
            print(' '.join(['#' if x == 1 else '.' for x in row]))
