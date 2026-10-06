"""
Программа загружает saved_model.npz и data_validation.csv,
затем выполняет прямое распространение и softmax и сохраняет прогнозы.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


EPS = 1e-12


def sigmoid(z):
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def activation_forward(z, name):
    if name == "sigmoid":
        return sigmoid(z)
    if name == "tanh":
        return np.tanh(z)
    if name == "relu":
        return np.maximum(0.0, z)
    raise ValueError(f"Неизвестная активация в модели: {name}")


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def binary_cross_entropy(y, probabilities):
    p_m = np.clip(probabilities[:, 1], EPS, 1.0 - EPS)
    y_float = y.astype(np.float64)
    return float(-np.mean(y_float * np.log(p_m) + (1.0 - y_float) * np.log(1.0 - p_m)))


def load_model(path):
    with np.load(path, allow_pickle=False) as data:
        layer_sizes = data["layer_sizes"].astype(int).tolist()
        activation = str(data["activation"].item())
        feature_mean = data["feature_mean"].astype(np.float64)
        feature_std = data["feature_std"].astype(np.float64)
        class_names = data["class_names"].astype(str)

        n_layers = len(layer_sizes) - 1
        weights = [data[f"W{i}"].astype(np.float64) for i in range(n_layers)]
        biases = [data[f"b{i}"].astype(np.float64) for i in range(n_layers)]

    return layer_sizes, activation, feature_mean, feature_std, class_names, weights, biases


def parse_row(row, line_no, expected_features):
    if len(row) == expected_features + 2 and row[1].strip() in ("B", "M"):
        sample_id = row[0].strip()
        label = 0 if row[1].strip() == "B" else 1
        feature_fields = row[2:]
    elif len(row) == expected_features + 1:
        sample_id = row[0].strip()
        label = None
        feature_fields = row[1:]
    elif len(row) == expected_features:
        sample_id = str(line_no)
        label = None
        feature_fields = row
    else:
        raise ValueError(
            f"Строка {line_no}: неподдерживаемое число столбцов {len(row)}; "
            f"ожидалось {expected_features}, {expected_features + 1} или {expected_features + 2}"
        )

    try:
        x = [float(v) for v in feature_fields]
    except ValueError as exc:
        raise ValueError(f"Строка {line_no}: признаки должны быть числами") from exc

    if not np.all(np.isfinite(x)):
        raise ValueError(f"Строка {line_no}: NaN/inf в признаках")

    return sample_id, label, x


def load_dataset(path, expected_features):
    ids = []
    labels = []
    X = []
    has_labels = None

    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for line_no, row in enumerate(reader, start=1):
            if not row:
                continue
            sample_id, label, x = parse_row(row, line_no, expected_features)

            row_has_label = label is not None
            if has_labels is None:
                has_labels = row_has_label
            elif has_labels != row_has_label:
                raise ValueError("Нельзя смешивать строки с метками и без меток в одном файле")

            ids.append(sample_id)
            X.append(x)
            if label is not None:
                labels.append(label)

    if not X:
        raise ValueError("Входной набор данных пуст")

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64) if has_labels else None
    return ids, X, y


def forward(X, activation, weights, biases):
    a = X
    for i in range(len(weights) - 1):
        z = a @ weights[i] + biases[i]
        a = activation_forward(z, activation)
    logits = a @ weights[-1] + biases[-1]
    return softmax(logits)


def save_predictions(path, ids, probabilities, class_names, y=None):
    predictions = np.argmax(probabilities, axis=1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["id", "predicted", "prob_B", "prob_M"]
        if y is not None:
            header.append("actual")
        writer.writerow(header)

        for i, sample_id in enumerate(ids):
            row = [
                sample_id,
                class_names[predictions[i]],
                f"{probabilities[i, 0]:.10f}",
                f"{probabilities[i, 1]:.10f}",
            ]
            if y is not None:
                row.append(class_names[y[i]])
            writer.writerow(row)


def print_confusion_matrix(y, pred):
    # строки: actual B/M, столбцы: predicted B/M.
    matrix = np.zeros((2, 2), dtype=int)
    for actual, predicted in zip(y, pred):
        matrix[actual, predicted] += 1

    print("Confusion matrix (строки=actual, столбцы=predicted):")
    print("            pred B   pred M")
    print(f"actual B   {matrix[0, 0]:6d}   {matrix[0, 1]:6d}")
    print(f"actual M   {matrix[1, 0]:6d}   {matrix[1, 1]:6d}")


def main():
    parser = argparse.ArgumentParser(description="Прогнозирование MLP")
    parser.add_argument("--model", default="saved_model.npz", help="Сохраненная модель .npz")
    parser.add_argument("--data", default="data_validation.csv", help="CSV для прогноза")
    parser.add_argument("--output", default="predictions.csv", help="Куда сохранить прогнозы")
    args = parser.parse_args()

    (
        layer_sizes,
        activation,
        feature_mean,
        feature_std,
        class_names,
        weights,
        biases,
    ) = load_model(args.model)

    expected_features = layer_sizes[0]
    ids, X, y = load_dataset(args.data, expected_features)

    if X.shape[1] != expected_features:
        raise ValueError(
            f"Модель ожидает {expected_features} признаков, получено {X.shape[1]}"
        )

    X = (X - feature_mean) / feature_std
    probabilities = forward(X, activation, weights, biases)
    predictions = np.argmax(probabilities, axis=1)

    save_predictions(args.output, ids, probabilities, class_names, y)

    print(f"Объектов: {len(X)}")
    print(f"Архитектура модели: {' -> '.join(map(str, layer_sizes))}")
    print(f"Скрытая активация: {activation}; выход: softmax")
    print(f"Прогнозы сохранены: {args.output}")

    if y is not None:
        loss = binary_cross_entropy(y, probabilities)
        acc = float(np.mean(predictions == y))
        print(f"binary cross-entropy: {loss:.6f}")
        print(f"accuracy: {acc:.4f}")
        print_confusion_matrix(y, predictions)
    else:
        print("Истинных меток в файле нет, поэтому loss/accuracy не вычисляются.")


if __name__ == "__main__":
    main()
