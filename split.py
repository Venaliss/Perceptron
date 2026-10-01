"""Разделение набора данных на обучающую и проверочную части.

Флаг --eda дополнительно выполняет предварительный анализ данных
(статистика и графики в папке figures).
"""
import argparse
import os

import numpy as np
import pandas as pd

CLASSES = ["B", "M"]
BASE = ["radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave_points", "symmetry",
        "fractal_dimension"]
NAMES = [b + s for s in ("_mean", "_se", "_worst") for b in BASE]


def load_csv(path):
    df = pd.read_csv(path, header=None)
    if str(df.iloc[0, 1]) not in CLASSES:  # файл с заголовком
        df = pd.read_csv(path)
        df.columns = range(df.shape[1])
    return df


def eda(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    df = df.copy()
    df.columns = ["id", "diagnosis"] + NAMES
    X = df[NAMES]
    y = (df["diagnosis"] == "M").astype(int)

    print("размер:", df.shape)
    print("пропусков:", int(df.isna().sum().sum()),
          "| дубликатов id:", int(df["id"].duplicated().sum()))
    vc = df["diagnosis"].value_counts()
    print("классы: B=%d (%.1f%%), M=%d (%.1f%%)" % (
        vc["B"], 100 * vc["B"] / len(df), vc["M"], 100 * vc["M"] / len(df)))
    print(X.describe().T[["mean", "std", "min", "max"]].head(10).round(4))
    print("min std = %.5f (%s), max std = %.1f (%s)" % (
        X.std().min(), X.std().idxmin(), X.std().max(), X.std().idxmax()))
    corr_y = X.corrwith(y).sort_values(ascending=False)
    print("топ-8 корреляций с меткой M:")
    print(corr_y.head(8).round(3))
    print("наименее связанные с меткой:")
    print(corr_y.abs().sort_values().head(5).round(3))
    c = X.corr().abs().to_numpy()
    iu = np.triu_indices_from(c, k=1)
    print("пар признаков с |corr| > 0.9: %d из %d" % (
        int((c[iu] > 0.9).sum()), len(iu[0])))

    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.bar(["B", "M"], [vc["B"], vc["M"]], color=["tab:blue", "tab:red"])
    for i, v in enumerate([vc["B"], vc["M"]]):
        ax.text(i, v + 4, str(v), ha="center")
    ax.set(ylabel="Число объектов", title="Баланс классов")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "class_balance.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(14, 5.5))
    for ax, n in zip(axes.ravel(), NAMES[:10]):
        ax.hist(X.loc[y == 0, n], bins=25, alpha=0.6, label="B", color="tab:blue")
        ax.hist(X.loc[y == 1, n], bins=25, alpha=0.6, label="M", color="tab:red")
        ax.set_title(n, fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0, 0].legend()
    fig.suptitle("Распределения признаков (mean) по классам")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "histograms.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(X.corr(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set(title="Корреляционная матрица признаков",
           xlabel="номер признака", ylabel="номер признака")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "correlation.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 8))
    s = corr_y.sort_values()
    ax.barh(s.index, s.values,
            color=["tab:red" if v > 0 else "tab:blue" for v in s])
    ax.tick_params(labelsize=7)
    ax.set(xlabel="корреляция с меткой (M=1)",
           title="Связь признаков с диагнозом")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "corr_with_target.png"), dpi=150)
    plt.close(fig)
    print("графики сохранены в '%s'" % out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="data.csv")
    p.add_argument("--valid_size", type=float, default=0.2,
                   help="доля проверочной выборки")
    p.add_argument("--seed", type=int, default=42,
                   help="seed для воспроизводимого разделения")
    p.add_argument("--train_out", default="data_training.csv")
    p.add_argument("--valid_out", default="data_valid.csv")
    p.add_argument("--eda", action="store_true",
                   help="выполнить анализ данных и построить графики")
    p.add_argument("--figures", default="figures")
    a = p.parse_args()

    df = load_csv(a.dataset)
    if a.eda:
        eda(df, a.figures)
    rng = np.random.default_rng(a.seed)
    labels = df.iloc[:, 1].to_numpy()
    valid_idx = []
    # стратифицированное разделение: доли классов сохраняются в обеих частях
    for c in CLASSES:
        idx = rng.permutation(np.where(labels == c)[0])
        valid_idx.extend(idx[:int(round(len(idx) * a.valid_size))])
    mask = np.zeros(len(df), dtype=bool)
    mask[valid_idx] = True
    train = df[~mask].sample(frac=1.0, random_state=a.seed)
    valid = df[mask].sample(frac=1.0, random_state=a.seed)
    train.to_csv(a.train_out, header=False, index=False)
    valid.to_csv(a.valid_out, header=False, index=False)
    for name, part in (("train", train), ("valid", valid)):
        vc = part.iloc[:, 1].value_counts()
        print("%s: %d строк (B=%d, M=%d)" % (name, len(part),
                                              vc.get("B", 0), vc.get("M", 0)))
    print("сохранено: %s, %s" % (a.train_out, a.valid_out))


if __name__ == "__main__":
    main()
