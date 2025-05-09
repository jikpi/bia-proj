import os

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ex10_lm.lm_nn import LogisticMapNN, EarlyStopping, train_epoch, validate, ModelCheckpoint
from ex10_lm.logistic_map import LogisticMap


def plot_bd(bifurcation_data, figsize=(12, 8), dpi=100, s=0.1, alpha=0.3, filename="bfcd.png"):
    title = "Logistic Map Bifurcation Diagram"
    plt.figure(figsize=figsize, dpi=dpi)

    points_x = []
    points_y = []

    for a, steady_states in bifurcation_data:
        for x in steady_states:
            points_x.append(a)
            points_y.append(x)

    plt.scatter(points_x, points_y, s=s, alpha=alpha, c='blue', marker='.')

    plt.title(title)
    plt.xlabel("(a)")
    plt.ylabel("(x)")
    plt.xlim(min(points_x), max(points_x))
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(os.path.join("Outputs", filename), dpi=dpi, bbox_inches='tight')
    plt.close()


def lm_solve():
    a_start = 1.0
    a_stop = 4.0
    a_samples = 1000
    iterations = 1000
    skip = 100
    x0 = 0.6

    lm = LogisticMap(iterations=iterations, skip=skip, x0=x0)

    # pole parametru 'a', od 'a_start do 'a_stop', s poctem 'a_samples'
    a_values = np.linspace(a_start, a_stop, a_samples)

    data = lm.generate(a_values)
    plot_bd(data, figsize=(12, 8), dpi=100, s=0.1, alpha=0.3)
    pass


def lm_solve_ann():
    a_start = 1.0
    a_stop = 4.0
    a_samples = 1000
    a_train_samples = 300
    iterations = 1300
    skip = 200
    x0 = 0.6
    a_instances = 50

    device = torch.device("cpu")

    lm = LogisticMap(iterations=iterations, skip=skip, x0=x0)

    # trenovaci dataset neni vyvazeny, umele se zvysi pocet vzorku po critical_a_value
    # (tohle zvysilo kvalitu modelu hlavne po tech ~3.5)
    critical_a_value = 3.3  # od tohoto 'a' bude v trenovacim datasetu vice vzorku
    samples_below_critical = a_train_samples // 2
    samples_above_critical = a_train_samples - samples_below_critical

    # a pro < critical_a_value
    a_values_low = np.linspace(a_start, critical_a_value, samples_below_critical, endpoint=False)

    # a pro > critical_a_value
    a_values_high = np.linspace(critical_a_value, a_stop, samples_above_critical, endpoint=True)

    a_train_values = np.concatenate((a_values_low, a_values_high))
    a_train_values = np.unique(a_train_values)

    print("Creating training dataset...")
    X, y = lm.create_dataset(a_train_values, a_instances=a_instances)

    X_flat = np.array([sample.flatten() for sample in X])

    # trenovaci a validacni dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X_flat, y, test_size=0.2, random_state=10
    )

    # normalizace dat
    X_scaler = MinMaxScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_val_scaled = X_scaler.transform(X_val)

    y_scaler = MinMaxScaler().fit(y_train.reshape(-1, 1))
    y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()

    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled).view(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val_scaled).view(-1, 1)

    # pytorch datasety
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    print("Creating neural network...")
    input_dim = X_train_scaled.shape[1]

    model = LogisticMapNN(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # early stopping a checkpoint
    early_stopping = EarlyStopping(patience=20)
    checkpoint_path = os.path.join("Outputs", "best_model.pt")
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss')

    history = {
        'loss': [],
        'val_loss': []
    }

    # trenovani
    print("Training model...")
    epochs = 100

    for epoch in range(epochs):
        # 1 epoch
        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)

        # validace
        val_loss, val_mae = validate(model, val_loader, criterion, device)

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        metrics = {
            'val_loss': val_loss,
            'val_mae': val_mae
        }

        model_checkpoint(metrics, model)

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # obnoveni vah nejlepsiho modelu
    print(f"Restoring best model weights (val_loss: {model_checkpoint.best_score:.4f})")
    model_checkpoint.restore_best_weights(model)

    # data pro inferenci/vizualizaci
    print("Creating data for visualization...")
    a_input_values = np.linspace(a_start, a_stop, a_samples)

    bifurcation_data_nn = []
    model.eval()

    # pro kazde 'a' z 'a_input_values' se vytvori vstupni sekvence
    for a_true_for_input_seq in a_input_values:
        # vstupni sekvence podle true 'a'
        x_values_input_seq = lm.iterate(a_true_for_input_seq, x0=x0)
        steady_states_input_seq = x_values_input_seq[skip:]

        # priprava vstupni sekvence a predikce 'a'
        sample = steady_states_input_seq.reshape(1, -1)
        sample_scaled = X_scaler.transform(sample)
        sample_tensor = torch.FloatTensor(sample_scaled).to(device)

        with torch.no_grad():
            predicted_a_scaled = model(sample_tensor).item()

        # inverzni transformace pro 'a' pro originalni rozsah
        predicted_a_unclipped = y_scaler.inverse_transform([[predicted_a_scaled]])[0][0]

        # clip na min/max hodnoty
        predicted_a = np.clip(predicted_a_unclipped, a_start, a_stop)

        # vygenerovani hodnot x pro 'predicted_a'
        x_values_for_plot = lm.iterate(predicted_a, x0=x0)
        steady_states_for_plot = x_values_for_plot[skip:]

        bifurcation_data_nn.append((predicted_a, steady_states_for_plot))

    print("Creating plot...")
    plot_bd(bifurcation_data_nn, figsize=(12, 8), dpi=100, s=0.1, alpha=0.3, filename="bfcd_nn.png")
