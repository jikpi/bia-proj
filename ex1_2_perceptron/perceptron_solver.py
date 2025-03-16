import random
import numpy as np
import matplotlib.pyplot as plt
from ex1_2_perceptron.perceptron_network import PerceptronNetwork


# funkce
def linear_function(x):
    return 3 * x + 2


# zjisteni zda je bod nad nebo pod primkou
def classify_point(x, y):
    line_y = linear_function(x)
    if y > line_y:
        return 1
    else:
        return -1


# vytvoreni datasetu
def generate_dataset(size, x_range=(-20, 20), y_range=(-20, 20)):
    data = []
    labels = []

    for _ in range(size):
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        data.append([x, y])
        labels.append(classify_point(x, y))

    return data, labels


# vypocet primky z vah a biasu perceptronu
def perceptron_line(x, weights, bias):
    return (-weights[0] * x - bias) / weights[1]


def test_task():
    # train data
    train_size = 10000
    train_data, train_labels = generate_dataset(train_size)

    # test data
    test_size = 200
    test_data, test_labels = generate_dataset(test_size)

    # natrenovani perceptronu
    print("Training:")
    perceptron = PerceptronNetwork(input_size=2)
    perceptron.train(data=train_data, truth=train_labels, learning_rate=0.01, epochs=500)

    # otestovani presnosti
    accuracy = perceptron.test_classification(test_data, test_labels)

    print(f"Weights: {perceptron.weights}")
    print(f"Bias: {perceptron.bias}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    if perceptron.weights[1] != 0:
        slope = -perceptron.weights[0] / perceptron.weights[1]
        intercept = -perceptron.bias / perceptron.weights[1]
        print(f"Decision boundary based on weights: y = {slope:.4f}x + {intercept:.4f}")



    # vizualizace
    predictions = [perceptron.predict(point) for point in test_data]

    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    predictions = np.array(predictions)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    scatter1 = ax1.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, cmap='coolwarm',
                           marker='o', edgecolors='k', s=50, alpha=0.7)
    ax1.set_title('Truth classifications')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    line_x = np.linspace(min(test_data[:, 0]), max(test_data[:, 0]), 100)
    line_y = [linear_function(x) for x in line_x]

    ax1.plot(line_x, line_y, 'g-', lw=2, label='True boundary')
    ax1.grid(True)
    ax1.legend()

    cbar1 = fig.colorbar(scatter1, ax=ax1, ticks=[-1, 1])
    cbar1.set_label('Class')
    cbar1.set_ticklabels(['Below', 'Above'])

    scatter2 = ax2.scatter(test_data[:, 0], test_data[:, 1], c=predictions, cmap='coolwarm',
                           marker='o', edgecolors='k', s=50, alpha=0.7)
    ax2.set_title('Perceptron Predictions')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    ax2.plot(line_x, line_y, 'g-', lw=2, label='True Line: y = 3x + 2')

    perceptron_y = [perceptron_line(x, perceptron.weights, perceptron.bias) for x in line_x]
    ax2.plot(line_x, perceptron_y, 'b--', lw=2, label='Decision Boundary')

    ax2.grid(True)
    ax2.legend()

    cbar2 = fig.colorbar(scatter2, ax=ax2, ticks=[-1, 1])
    cbar2.set_label('Predicted Class')
    cbar2.set_ticklabels(['Below', 'Above'])

    fig.suptitle(f'Accuracy: {accuracy * 100:.2f}%',
                 fontsize=16)

    ax1.set_xlim([min(test_data[:, 0]), max(test_data[:, 0])])
    ax1.set_ylim([min(test_data[:, 1]), max(test_data[:, 1])])
    ax2.set_xlim([min(test_data[:, 0]), max(test_data[:, 0])])
    ax2.set_ylim([min(test_data[:, 1]), max(test_data[:, 1])])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'Outputs/perceptron_graph.png', dpi=300, bbox_inches='tight')
    plt.show()
