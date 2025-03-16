import random
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from ex1_2_perceptron.neural_network import NeuralNetwork


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def xor_function(x, y):
    return 1 if (x == 1 and y == 0) or (x == 0 and y == 1) else 0


def xor_data_domain():
    data = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    labels = [[xor_function(x, y)] for x, y in data]

    return data, labels


def xor_create_dataset(size):
    data = []
    labels = []

    for _ in range(size):
        x = random.randint(0, 1)
        y = random.randint(0, 1)

        xor_result = xor_function(x, y)

        data.append([x, y])
        labels.append([xor_result])

    return data, labels


def visualize_xor_nn(nn, resolution=100):
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))

    Z = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = nn.classify_binary([xx[i, j], yy[i, j]])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    true_colors = ListedColormap(['#FFAAAA', '#AAAAFF'])
    ax1.set_title('True XOR Function')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    true_Z = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            true_Z[i, j] = xor_function(round(xx[i, j]), round(yy[i, j]))

    ax1.contourf(xx, yy, true_Z, cmap=true_colors, alpha=0.8)

    xor_data, xor_labels = xor_data_domain()
    xor_data = np.array(xor_data)
    xor_labels = np.array(xor_labels).flatten()

    ax1.scatter(xor_data[:, 0], xor_data[:, 1], c=xor_labels,
                cmap=ListedColormap(['#FF0000', '#0000FF']),
                edgecolors='k', s=150, marker='o')

    for i, (x, y) in enumerate(xor_data):
        ax1.text(x, y, f"({int(x)}, {int(y)}): {int(xor_labels[i])}",
                 ha='center', va='bottom', fontsize=12)

    ax1.grid(True)
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([y_min, y_max])

    pred_colors = ListedColormap(['#FFAAAA', '#AAAAFF'])
    ax2.set_title('Neural Network Predictions')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    ax2.contourf(xx, yy, Z, cmap=pred_colors, alpha=0.8)

    ax2.scatter(xor_data[:, 0], xor_data[:, 1], c=xor_labels,
                cmap=ListedColormap(['#FF0000', '#0000FF']),
                edgecolors='k', s=150, marker='o')

    for i, (x, y) in enumerate(xor_data):
        prediction = nn.classify_binary([x, y])
        confidence = nn.infer([x, y])[0]

        ax2.text(x, y, f"({int(x)}, {int(y)}): {prediction}\n({confidence:.2f})",
                 ha='center', va='bottom', fontsize=12)

    ax2.grid(True)
    ax2.set_xlim([x_min, x_max])
    ax2.set_ylim([y_min, y_max])

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=pred_colors), cax=cbar_ax)
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(['0', '1'])
    cbar.set_label('XOR Output')

    fig.suptitle('XOR: Truth vs Prediction', fontsize=16)

    return fig


def test_task():
    train_size = 1000
    train_data, train_labels = xor_create_dataset(train_size)

    # 2 vstupy, 3 skryte neurony, 1 vystup
    # se 2 skrytymi neurony se obcas sekne v lokalnim minimu
    nn = NeuralNetwork(input_size=2, structure=[3, 1], activation_function=sigmoid,
                       activation_derivative=sigmoid_derivative)

    print(f"Training ({train_size}):")
    nn.train(train_data, train_labels, learning_rate=0.1, epochs=500)

    print("\nTesting on the complete domain of XOR:")
    domain_data, domain_labels = xor_data_domain()
    base_accuracy, base_results = nn.test_binary(domain_data, domain_labels)
    print(f"Accuracy: {base_accuracy:.2f}")

    fig = visualize_xor_nn(nn)

    plt.savefig(f'Outputs/xor_neural_network_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
