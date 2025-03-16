from collections.abc import Callable

from ex1_2_perceptron.perceptron import Perceptron


class NeuralNetwork:
    def __init__(self, input_size: int,  # pocet vstupu
                 structure: list[int],
                 # list, kde cisla reprezentuji pocet neuronu v kazde vrstve. vrstvy jsou plne propojeny
                 activation_function: callable,
                 activation_derivative: Callable,  # derivace aktivacni funkce
                 ):

        self.input_size = input_size
        self.structure = structure
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.layers = []

        # vstup do prvni vrstvy
        first_layer = [Perceptron(input_size) for _ in range(structure[0])]
        self.layers.append(first_layer)

        # ostatni vrstvy a vystup
        for i in range(1, len(structure)):
            prev_layer_size = structure[i - 1]
            current_layer = [Perceptron(prev_layer_size) for _ in range(structure[i])]
            self.layers.append(current_layer)

    # inference s danym vstupem
    def infer(self, inputs):
        # pruchod vsemi vrstvami, vystup predesle vrstvy je vstup pro dalsi vrstvu
        layer_result = inputs

        for layer in self.layers:
            layer_outputs = []
            for perceptron in layer:
                output = perceptron.infer(layer_result, self.activation_function)
                layer_outputs.append(output)
            layer_result = layer_outputs

        return layer_result

    def train(self, training_data,  # list vstupnich vektoru pro trenink
              expected_outputs,  # list ocekavanych vystupnich vektoru, korespondujicich ke kazdemu vstupnimu vektoru
              learning_rate=0.1, epochs=1000
              ):

        for epoch in range(epochs):
            total_error = 0

            for i in range(len(training_data)):

                # --------------
                # Inference pro dany vstup
                # --------------
                current_input = training_data[i]
                expected_output = expected_outputs[i]

                # Aktivace (vystupy dane vrstvy (tedy kazdeho perceptronu)) jsou ulozeny
                activations = [current_input]  # jako prvni aktivace je vstup
                layer_result = current_input  # vystup predchozi vrstvy, ktery se posila do dalsi vrstvy. zacina jako vstup

                # pruchod vstupu vsemi vrstvami
                for layer in self.layers:
                    layer_outputs = []  # list vystupnich vektoru pro kazdy perceptron v dane vrstve
                    for perceptron in layer:
                        output = perceptron.infer(layer_result, self.activation_function)
                        layer_outputs.append(output)
                    layer_result = layer_outputs
                    activations.append(layer_outputs)

                # vypocet vystupni chyby na zaklade ocekavaneho vystupu
                output_errors = []
                for j in range(len(expected_output)):
                    error = expected_output[j] - activations[-1][j]
                    output_errors.append(error)
                    total_error += error ** 2  # MSE

                # --------------
                # Backpropagation
                # --------------
                all_deltas = [output_errors]  # podil kazdeho neuronu na chybe v posledni vrstve

                # iterace od posledni vrstvy k prvni bez vstupni vrstvy
                for l in range(len(self.layers) - 1, 0, -1):
                    layer_deltas = []
                    next_layer = self.layers[l]

                    # iterace pro kazdy neuron v predchozi vrstve a vypocet jeho podilu na chybe
                    for j in range(len(self.layers[l - 1])):
                        error = 0.0
                        for k in range(len(next_layer)):
                            error += all_deltas[0][k] * next_layer[k].weights[j]

                        output = activations[l][j]
                        derivative = self.activation_derivative(output)
                        delta = error * derivative  # o kolik se ma zmenit vahovy koeficient a bias daneho neuronu
                        layer_deltas.append(delta)

                    # pridani delty vrstvy na zacatek
                    all_deltas.insert(0, layer_deltas)

                # uprava vah a bias v perceptronech
                for l in range(len(self.layers)):
                    layer = self.layers[l]
                    prev_activations = activations[l]

                    # pro kazdy neuron v dane vrstve
                    for j in range(len(layer)):
                        # pro kazdou vahu neuronu
                        for k in range(len(layer[j].weights)):
                            layer[j].weights[k] += (learning_rate * all_deltas[l][j]  # velikost a směr zmeny
                                                    * prev_activations[k]  # proporcionalne k velikosti vstupu
                                                    )
                        layer[j].bias += learning_rate * all_deltas[l][j]

            if epoch % 100 == 0:
                mse = total_error / len(training_data)
                print(f"Epoch {epoch}, MSE: {mse:.6f}")

    def classify_binary(self, inputs):
        outputs = self.infer(inputs)
        return 1 if outputs[0] >= 0.5 else 0

    def test_binary(self, test_data,  # list vstupnich vektoru pro test
                    expected_outputs  # list pravdivych vystupu pro vstup
                    ):

        correct = 0
        total = len(test_data)

        results = []
        predictions = []

        for i in range(total):

            # binarni klasifikace
            raw_output = self.infer(test_data[i])
            prediction = 1 if raw_output[0] >= 0.5 else 0

            # porovnani s ocekavanym vystupem
            expected = expected_outputs[i][0]

            predictions.append(prediction)
            results.append((test_data[i], raw_output, prediction, expected))

            if prediction == expected:
                correct += 1

        accuracy = correct / total
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Matched: {correct} / {total}")

        print("\nLast 10 test results:")
        count = 0
        for inputs, raw_output, prediction, expected in results:
            print(
                f"Input: {inputs}, Raw Output: {[f'{o:.4f}' for o in raw_output]}, Prediction: {prediction}, Expected: {expected}")

            count += 1
            if count >= 10:
                break

        return accuracy, results
