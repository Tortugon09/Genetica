import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


error_norm_epochs = []
weight_evolution = []
initial_weights = None
final_weights = None
num_epochs = 0
permissible_error = 0


def train_perceptron(learning_rate, epochs, file_path, progress_bar):
    global error_norm_epochs, weight_evolution, initial_weights, num_epochs, final_weights

    error_norm_epochs, weight_evolution = [], []
    data_frame = pd.read_csv(file_path, delimiter=';', header=None)
    num_features = data_frame.shape[1] - 1

    weights = np.random.uniform(0, 1, (num_features + 1, 1)).round(4)
    x_columns = np.hstack([data_frame.iloc[:, :-1].values, np.ones((data_frame.shape[0], 1))])
    y_column = data_frame.iloc[:, -1].values.reshape(-1, 1)

    initial_weights = weights.copy()
    num_epochs = epochs
    print(f"Initial weights: {weights.T}")

    for _ in range(num_features + 1):
        weight_evolution.append([])

    for epoch in range(epochs):
        calculated_y = np.where(x_columns @ weights >= 0, 1, 0).reshape(-1, 1)
        errors = y_column - calculated_y
        error_norm_epochs.append(np.linalg.norm(errors))

        for i in range(num_features + 1):
            weight_evolution[i].append(weights[i, 0])

        weights += learning_rate * (x_columns.T @ errors)
        weights = np.round(weights, 4)

        progress_bar['value'] = (epoch + 1) / epochs * 100
        progress_bar.update()

    final_weights = weights


def display_results():

    plt.style.use('seaborn-v0_8') 
    global error_norm_epochs, weight_evolution
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(error_norm_epochs) + 1), error_norm_epochs)
    plt.title('Evolución del Error Absoluto(|e|)')
    plt.xlabel('Épocas')
    plt.ylabel('Error Absoluto(|e|)')

    plt.subplot(1, 2, 2)
    for i, weight_epoch in enumerate(weight_evolution):
        plt.plot(range(1, len(weight_epoch) + 1), weight_epoch, label=f'W {i + 1}')
    plt.title('Evolución de Pesos (W)')
    plt.xlabel('Epocas')
    plt.ylabel('Valor de W')
    plt.legend()

    plt.tight_layout()
    plt.show()

def get_weights():
    return initial_weights, final_weights, num_epochs, permissible_error
