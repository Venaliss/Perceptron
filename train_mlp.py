"""Обучение многослойного персептрона

По умолчанию используются data_training.csv и data_validation.csv, а сеть имеет
архитектуру 30 -> 24 -> 24 -> 2, то есть два скрытых слоя.

На выходе используется softmax. Для двух классов B/M бинарная кросс-энтропия
по вероятности класса M совпадает с кросс-энтропией по двум softmax-выходам:
            BCE = -[y log(p_M) + (1-y) log(p_B)].
Для softmax + cross-entropy градиент по логитам равен (p - y_one_hot).
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


N_COLUMNS = 32
N_FEATURES = 30
CLASS_NAMES = np.array(["B", "M"])
EPS = 1e-12


def load_labeled_csv(path: str):
    X = []
    y = []

    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for line_no, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) != N_COLUMNS:
                raise ValueError(
                    f"{path}, строка {line_no}: ожидалось {N_COLUMNS} столбца, получено {len(row)}"
                )
            label = row[1].strip()
            if label not in ("B", "M"):
                raise ValueError(f"{path}, строка {line_no}: диагноз должен быть B или M")
            try:
                values = [float(v) for v in row[2:]]
            except ValueError as exc:
                raise ValueError(f"{path}, строка {line_no}: признаки должны быть числами") from exc
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{path}, строка {line_no}: NaN/inf в признаках")
            X.append(values)
            y.append(0 if label == "B" else 1)

    if not X:
        raise ValueError(f"{path}: пустой набор данных")

    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def one_hot(y, n_classes=2):
    out = np.zeros((len(y), n_classes), dtype=np.float64)
    out[np.arange(len(y)), y] = 1.0
    return out


def sigmoid(z):
    # устойчивая сигмоида без переполнения exp для больших |z|
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def activation_forward(z, name: str):
    if name == "sigmoid":
        return sigmoid(z)
    if name == "tanh":
        return np.tanh(z)
    if name == "relu":
        return np.maximum(0.0, z)
    raise ValueError(f"Неизвестная активация: {name}")


def activation_derivative(z, a, name: str):
    if name == "sigmoid":
        return a * (1.0 - a)
    if name == "tanh":
        return 1.0 - a * a
    if name == "relu":
        return (z > 0.0).astype(np.float64)
    raise ValueError(f"Неизвестная активация: {name}")


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def binary_cross_entropy(y, probabilities):
    # BCE по вероятности класса M (1)
    p_m = np.clip(probabilities[:, 1], EPS, 1.0 - EPS)
    y_float = y.astype(np.float64)
    return float(-np.mean(y_float * np.log(p_m) + (1.0 - y_float) * np.log(1.0 - p_m)))


def accuracy(y, probabilities):
    predictions = np.argmax(probabilities, axis=1)
    return float(np.mean(predictions == y))


class MLP:
    def __init__(self, layer_sizes, activation="sigmoid", seed=42):
        if len(layer_sizes) < 4:
            raise ValueError("Нужны входной, как минимум два скрытых и выходной слои")
        if layer_sizes[-1] != 2:
            raise ValueError("Для B/M выходной слой должен содержать 2 нейрона")

        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        self.rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []

        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            if activation == "relu" and fan_out != layer_sizes[-1]:
                # He initialization для ReLU.
                W = self.rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
            else:
                # Xavier/Glorot uniform
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                W = self.rng.uniform(-limit, limit, size=(fan_in, fan_out))
            b = np.zeros((1, fan_out), dtype=np.float64)
            self.weights.append(W.astype(np.float64))
            self.biases.append(b)

    def forward(self, X, keep_cache=False):
        activations = [X]
        pre_activations = []
        a = X

        # все слои, кроме последнего: выбранная скрытая активация
        for i in range(len(self.weights) - 1):
            z = a @ self.weights[i] + self.biases[i]
            a = activation_forward(z, self.activation)
            pre_activations.append(z)
            activations.append(a)

        # выходной слой: softmax
        logits = a @ self.weights[-1] + self.biases[-1]
        probabilities = softmax(logits)
        pre_activations.append(logits)
        activations.append(probabilities)

        if keep_cache:
            return probabilities, activations, pre_activations
        return probabilities

    def backward(self, X_batch, y_batch):
        probabilities, activations, pre_activations = self.forward(X_batch, keep_cache=True)
        y_oh = one_hot(y_batch, 2)
        batch_size = len(X_batch)

        grad_weights = [None] * len(self.weights)
        grad_biases = [None] * len(self.biases)

        # softmax + CE/BCE для двух классов: dL/dz = (p - y_one_hot) / N
        delta = (probabilities - y_oh) / batch_size

        # градиенты выходного слоя.
        grad_weights[-1] = activations[-2].T @ delta
        grad_biases[-1] = np.sum(delta, axis=0, keepdims=True)

        # обратное распространение через скрытые слои
        for layer in range(len(self.weights) - 2, -1, -1):
            delta = delta @ self.weights[layer + 1].T
            z = pre_activations[layer]
            a = activations[layer + 1]
            delta *= activation_derivative(z, a, self.activation)

            grad_weights[layer] = activations[layer].T @ delta
            grad_biases[layer] = np.sum(delta, axis=0, keepdims=True)

        return grad_weights, grad_biases

    def step(self, grad_weights, grad_biases, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grad_weights[i]
            self.biases[i] -= learning_rate * grad_biases[i]


def standardize_train_valid(X_train, X_valid):
    # параметры стандартизации считаются только по train,
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (X_train - mean) / std, (X_valid - mean) / std, mean, std


def train_model(model, X_train, y_train, X_valid, y_valid, epochs, batch_size, learning_rate, seed):
    rng = np.random.default_rng(seed)
    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_acc": [],
        "valid_acc": [],
    }

    n = len(X_train)

    for epoch in range(1, epochs + 1):
        indices = rng.permutation(n)

        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            grad_w, grad_b = model.backward(X_batch, y_batch)
            model.step(grad_w, grad_b, learning_rate)

        train_probs = model.forward(X_train)
        valid_probs = model.forward(X_valid)

        train_loss = binary_cross_entropy(y_train, train_probs)
        valid_loss = binary_cross_entropy(y_valid, valid_probs)
        train_acc = accuracy(y_train, train_probs)
        valid_acc = accuracy(y_valid, valid_probs)

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["train_acc"].append(train_acc)
        history["valid_acc"].append(valid_acc)

        print(
            f"epoch {epoch:03d}/{epochs:03d} - "
            f"loss: {train_loss:.6f} - acc: {train_acc:.4f} - "
            f"val_loss: {valid_loss:.6f} - val_acc: {valid_acc:.4f}"
        )

    return history


def save_model(path, model, feature_mean, feature_std):
    payload = {
        "layer_sizes": np.asarray(model.layer_sizes, dtype=np.int64),
        "activation": np.asarray(model.activation),
        "feature_mean": feature_mean.astype(np.float64),
        "feature_std": feature_std.astype(np.float64),
        "class_names": CLASS_NAMES,
    }
    for i, (W, b) in enumerate(zip(model.weights, model.biases)):
        payload[f"W{i}"] = W
        payload[f"b{i}"] = b

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print(f"Модель сохранена: {path}")


def plot_history(history, output_path, show=True):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="train loss")
    axes[0].plot(epochs, history["valid_loss"], label="validation loss")
    axes[0].set_title("Кривая функции потерь")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train accuracy")
    axes[1].plot(epochs, history["valid_acc"], label="validation accuracy")
    axes[1].set_title("Кривая точности")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Кривые обучения сохранены: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Обучение MLP с нуля")
    parser.add_argument("--train", default="data_training.csv", help="Обучающий CSV")
    parser.add_argument("--valid", default="data_validation.csv", help="Проверочный CSV")
    parser.add_argument("--model", default="saved_model.npz", help="Куда сохранить модель")
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[24, 24],
        help="Размеры скрытых слоев, например: --layers 24 24 24",
    )
    parser.add_argument(
        "--activation",
        choices=["sigmoid", "tanh", "relu"],
        default="sigmoid",
        help="Активация скрытых слоев",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", default="learning_curves.png")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Не открывать интерактивное окно Matplotlib",
    )
    args = parser.parse_args()

    if len(args.layers) < 2:
        parser.error("По условию задания нужно как минимум два скрытых слоя")
    if any(size <= 0 for size in args.layers):
        parser.error("Размер каждого скрытого слоя должен быть положительным")
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("epochs, batch-size и learning-rate должны быть > 0")

    X_train, y_train = load_labeled_csv(args.train)
    X_valid, y_valid = load_labeled_csv(args.valid)

    if X_train.shape[1] != N_FEATURES or X_valid.shape[1] != N_FEATURES:
        raise ValueError(f"Ожидалось {N_FEATURES} признаков")

    X_train, X_valid, feature_mean, feature_std = standardize_train_valid(X_train, X_valid)

    layer_sizes = [N_FEATURES] + args.layers + [2]
    print(f"X_train shape: {X_train.shape}")
    print(f"X_valid shape: {X_valid.shape}")
    print(f"Архитектура: {' -> '.join(map(str, layer_sizes))}")
    print(f"Скрытая активация: {args.activation}; выход: softmax")
    print(f"seed = {args.seed}")

    model = MLP(layer_sizes, activation=args.activation, seed=args.seed)
    history = train_model(
        model,
        X_train,
        y_train,
        X_valid,
        y_valid,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    save_model(args.model, model, feature_mean, feature_std)
    plot_history(history, args.plot, show=not args.no_show)


if __name__ == "__main__":
    main()
