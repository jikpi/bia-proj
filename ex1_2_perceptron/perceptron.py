import random
from collections.abc import Callable


class Perceptron:
    def __init__(self, input_size):
        # vahy a bias
        self.weights = [0.0] * input_size
        for i in range(input_size):
            self.weights[i] = (2 * random.random()) - 1

        self.bias = random.random() * 2 - 1

    # inference perceptronu s danou aktivacni funkci
    def infer(self, inputs, activation_function: Callable):
        weighted_sum = self.bias
        for i in range(len(inputs)):
            weighted_sum += inputs[i] * self.weights[i]

        return activation_function(weighted_sum)
