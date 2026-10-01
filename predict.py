"""Прогнозирование обученной моделью и оценка бинарной кросс-энтропией."""
import argparse

import numpy as np

from train import CLASSES, MLP, binary_crossentropy, load_csv, one_hot, split_xy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="saved_model.npy")
    p.add_argument("--dataset", default="data_valid.csv")
    p.add_argument("--output", help="CSV с предсказаниями (необязательно)")
    a = p.parse_args()

    net, extra = MLP.load(a.model)
    df = load_csv(a.dataset)
    x, y = split_xy(df)
    x = (x - np.array(extra["mean"])) / np.array(extra["std"])

    proba = net.predict_proba(x)          # (N, 2): P(B), P(M)
    pred = proba.argmax(axis=1)
    bce = binary_crossentropy(proba, one_hot(y))
    acc = np.mean(pred == y)

    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    print("samples: %d" % len(y))
    print("binary cross-entropy: %.4f" % bce)
    print("accuracy: %.4f" % acc)
    print("precision (M): %.4f  recall (M): %.4f  F1: %.4f" % (prec, rec, f1))
    print("confusion matrix (строки - истинный класс, столбцы - предсказание)")
    print("        pred B  pred M")
    print("true B  %6d  %6d" % (tn, fp))
    print("true M  %6d  %6d" % (fn, tp))

    if a.output:
        out = df.iloc[:, :2].copy()
        out.columns = ["id", "true"]
        out["pred"] = [CLASSES[i] for i in pred]
        out["p_malignant"] = proba[:, 1].round(6)
        out.to_csv(a.output, index=False)
        print("предсказания сохранены в '%s'" % a.output)


if __name__ == "__main__":
    main()
