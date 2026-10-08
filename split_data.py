import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


N_COLUMNS = 32
N_FEATURES = 30
VALID_LABELS = {"B", "M"}


def read_dataset(path: str):
    rows = []
    features = []
    labels = []
    ids = []

    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for line_no, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) != N_COLUMNS:
                raise ValueError(
                    f"Строка {line_no}: ожидалось {N_COLUMNS} столбца, получено {len(row)}"
                )

            diagnosis = row[1].strip()
            if diagnosis not in VALID_LABELS:
                raise ValueError(
                    f"Строка {line_no}: неизвестный диагноз {diagnosis!r}; ожидался B или M"
                )

            try:
                x = [float(value) for value in row[2:]]
            except ValueError as exc:
                raise ValueError(f"Строка {line_no}: признаки должны быть числами") from exc

            if not np.all(np.isfinite(x)):
                raise ValueError(f"Строка {line_no}: найдено NaN/inf в признаках")

            rows.append(row)
            ids.append(row[0].strip())
            labels.append(diagnosis)
            features.append(x)

    if not rows:
        raise ValueError("Файл пуст")

    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    return rows, ids, X, y


def print_statistics(ids, X, y):
    count_b = int(np.sum(y == "B"))
    count_m = int(np.sum(y == "M"))
    n = len(y)

    feature_means = X.mean(axis=0)
    feature_stds = X.std(axis=0)
    feature_mins = X.min(axis=0)
    feature_maxs = X.max(axis=0)

    print("Статистический анализ")
    print(f"Объектов: {n}")
    print(f"Числовых признаков: {X.shape[1]}")
    print(f"B (benign): {count_b} ({100.0 * count_b / n:.2f}%)")
    print(f"M (malignant): {count_m} ({100.0 * count_m / n:.2f}%)")
    print(f"Уникальных значений первого столбца: {len(set(ids))} из {n}")
    print("Первый столбец поэтому рассматривается как идентификатор и в MLP не подается.")
    print(
        "Масштабы признаков сильно различаются: "
        f"std от {feature_stds.min():.6g} до {feature_stds.max():.6g}; "
        f"общий диапазон значений от {feature_mins.min():.6g} до {feature_maxs.max():.6g}."
    )
    print()
    print("Первые 5 признаков (mean / std / min / max):")
    for i in range(min(5, X.shape[1])):
        print(
            f"  feature_{i + 1:02d}: "
            f"{feature_means[i]:.6g} / {feature_stds[i]:.6g} / "
            f"{feature_mins[i]:.6g} / {feature_maxs[i]:.6g}"
        )


def save_analysis_plot(X, y, output_path: str):
    # корреляция с бинарной меткой
    target = (y == "M").astype(np.float64)
    correlations = []
    for j in range(X.shape[1]):
        x = X[:, j]
        if np.std(x) < 1e-15:
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(x, target)[0, 1]))
    correlations = np.asarray(correlations)

    stds = X.std(axis=0)
    indices = np.arange(1, X.shape[1] + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].bar(["B", "M"], [np.sum(y == "B"), np.sum(y == "M")])
    axes[0].set_title("Баланс классов")
    axes[0].set_xlabel("Диагноз")
    axes[0].set_ylabel("Количество")

    axes[1].bar(indices, stds)
    axes[1].set_yscale("log")
    axes[1].set_title("Разные масштабы признаков")
    axes[1].set_xlabel("Номер признака")
    axes[1].set_ylabel("Стандартное отклонение (log scale)")

    axes[2].bar(indices, correlations)
    axes[2].axhline(0.0, linewidth=0.8)
    axes[2].set_title("Корреляция признаков с классом M")
    axes[2].set_xlabel("Номер признака")
    axes[2].set_ylabel("Корреляция")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"График анализа сохранен: {output_path}")


def stratified_split(rows, y, train_ratio: float, seed: int):
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio должен быть между 0 и 1")

    rng = np.random.default_rng(seed)
    train_indices = []
    valid_indices = []

    # разбиваем каждый класс отдельно
    for label in ("B", "M"):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)

        n_train = int(round(len(idx) * train_ratio))
        n_train = min(max(n_train, 1), len(idx) - 1)

        train_indices.extend(idx[:n_train].tolist())
        valid_indices.extend(idx[n_train:].tolist())

    # перемешиваем объединенные части
    train_indices = np.asarray(train_indices, dtype=int)
    valid_indices = np.asarray(valid_indices, dtype=int)
    rng.shuffle(train_indices)
    rng.shuffle(valid_indices)

    train_rows = [rows[i] for i in train_indices]
    valid_rows = [rows[i] for i in valid_indices]
    return train_rows, valid_rows


def write_rows(path: str, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def count_classes(rows):
    labels = [row[1] for row in rows]
    return labels.count("B"), labels.count("M")


def find_default_input():
    preferred = Path("data.csv")
    if preferred.exists():
        return str(preferred)

    excluded = {
        "data_training.csv",
        "data_validation.csv",
        "predictions.csv",
    }
    candidates = sorted(
        p for p in Path(".").glob("data*.csv")
        if p.name not in excluded
    )

    if len(candidates) == 1:
        print(f"data.csv не найден; использую {candidates[0]}")
        return str(candidates[0])
    if not candidates:
        raise FileNotFoundError(
            "Не найден data.csv. Положите исходный CSV рядом со скриптом "
        )
    raise FileNotFoundError(
        "Найдено несколько подходящих data*.csv."
    )


def main():
    parser = argparse.ArgumentParser(description="Воспроизводимое разбиение")
    parser.add_argument(
        "--input",
        default=None,
        help="Исходный CSV.",
    )
    parser.add_argument("--train", default="data_training.csv", help="Выходной train CSV")
    parser.add_argument("--valid", default="data_validation.csv", help="Выходной validation CSV")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Доля train, по умолчанию 0.8")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости")
    parser.add_argument(
        "--analysis-plot",
        default="data_analysis.png",
        help="Файл с визуальным анализом",
    )
    args = parser.parse_args()

    input_path = args.input if args.input is not None else find_default_input()

    rows, ids, X, y = read_dataset(input_path)
    print_statistics(ids, X, y)
    save_analysis_plot(X, y, args.analysis_plot)

    train_rows, valid_rows = stratified_split(rows, y, args.train_ratio, args.seed)
    write_rows(args.train, train_rows)
    write_rows(args.valid, valid_rows)

    train_b, train_m = count_classes(train_rows)
    valid_b, valid_m = count_classes(valid_rows)

    print("\nРазбиение")
    print(f"seed = {args.seed}")
    print(f"train: {len(train_rows)} строк (B={train_b}, M={train_m}) -> {args.train}")
    print(f"valid: {len(valid_rows)} строк (B={valid_b}, M={valid_m}) -> {args.valid}")


if __name__ == "__main__":
    main()
