import random


class PerceptronNetwork:
    def __init__(self, input_size):
        # inicializace vah a bias v rozmezi [-1, 1]
        self.input_size = input_size
        self.weights = [(2 * random.random()) - 1 for _ in range(input_size)]
        self.bias = random.random() * 2 - 1

    # signum
    def signum(self, x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    # inference
    def predict(self, inputs):
        weighted_sum = self.bias
        for i in range(len(inputs)):
            weighted_sum += inputs[i] * self.weights[i]

        return self.signum(weighted_sum)

    # trenovani
    def train(self, data, truth, learning_rate=0.01, epochs=1000):
        for epoch in range(epochs):
            total_error = 0

            for i in range(len(data)):
                # vypocet predikce
                raw_prediction = self.predict(data[i])

                # vypocet chyby
                error = truth[i] - raw_prediction
                total_error += error ** 2

                # uprava vah: w_new = w_i + Error * input * learning_rate
                for j in range(len(self.weights)):
                    self.weights[j] += error * data[i][j] * learning_rate

                # aktualizace bias
                self.bias += error * learning_rate

            if epoch % 50 == 0:
                mse = total_error / len(data)
                print(f"Epoch {epoch}, MSE: {mse:.6f}")

    def test_classification(self, data, truth):
        correct = 0
        total = len(data)

        for i in range(total):
            prediction = self.predict(data[i])

            if prediction == truth[i]:
                correct += 1

        accuracy = correct / total
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Matched: {correct} / {total}")

        return accuracy
