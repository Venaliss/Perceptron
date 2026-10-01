"""Обучение многослойного персептрона (программа обучения).

Файл содержит и реализацию сети (активации, слои, потери, MLP), которую
использует predict.py.

Пример:
  python train.py --layer 24 24 24 --epochs 84 --loss binaryCrossentropy \
      --batch_size 8 --learning_rate 0.0314
Параметры можно также задать JSON-файлом: --config config.json
Проверка обратного распространения: python train.py --gradcheck
"""
import argparse
import json

import numpy as np

EPS = 1e-12

# ---------------------------------------------------------------- активации


def sigmoid_f(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_b(z, a, g):
    return g * a * (1.0 - a)


def relu_f(z):
    return np.maximum(z, 0.0)


def relu_b(z, a, g):
    return g * (z > 0)


def tanh_f(z):
    return np.tanh(z)


def tanh_b(z, a, g):
    return g * (1.0 - a ** 2)


def linear_f(z):
    return z


def linear_b(z, a, g):
    return g


def softmax_f(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def softmax_b(z, a, g):
    # Произведение якобиана softmax на градиент: a * (g - sum(g * a))
    return a * (g - (g * a).sum(axis=1, keepdims=True))


ACTIVATIONS = {
    "sigmoid": (sigmoid_f, sigmoid_b),
    "relu": (relu_f, relu_b),
    "tanh": (tanh_f, tanh_b),
    "linear": (linear_f, linear_b),
    "softmax": (softmax_f, softmax_b),
}

# ------------------------------------------------------------- инициализация


def init_weights(name, n_in, n_out, rng):
    if name == "heUniform":
        lim = np.sqrt(6.0 / n_in)
        return rng.uniform(-lim, lim, (n_in, n_out))
    if name == "heNormal":
        return rng.normal(0.0, np.sqrt(2.0 / n_in), (n_in, n_out))
    if name == "xavierUniform":
        lim = np.sqrt(6.0 / (n_in + n_out))
        return rng.uniform(-lim, lim, (n_in, n_out))
    if name == "xavierNormal":
        return rng.normal(0.0, np.sqrt(2.0 / (n_in + n_out)), (n_in, n_out))
    if name == "normal":
        return rng.normal(0.0, 0.05, (n_in, n_out))
    raise ValueError("неизвестный инициализатор: " + name)


# -------------------------------------------------------------------- потери


def binary_crossentropy(p, y):
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def binary_crossentropy_grad(p, y):
    p = np.clip(p, EPS, 1.0 - EPS)
    return (-(y / p) + (1.0 - y) / (1.0 - p)) / y.size


def categorical_crossentropy(p, y):
    p = np.clip(p, EPS, 1.0)
    return float(-np.mean(np.sum(y * np.log(p), axis=1)))


def categorical_crossentropy_grad(p, y):
    p = np.clip(p, EPS, 1.0)
    return -(y / p) / y.shape[0]


LOSSES = {
    "binaryCrossentropy": (binary_crossentropy, binary_crossentropy_grad),
    "categoricalCrossentropy": (categorical_crossentropy,
                                categorical_crossentropy_grad),
}

# --------------------------------------------------------------------- слой


class DenseLayer:
    def __init__(self, units, activation="sigmoid",
                 weights_initializer="heUniform"):
        if activation not in ACTIVATIONS:
            raise ValueError("неизвестная активация: " + activation)
        self.units = units
        self.activation = activation
        self.initializer = weights_initializer
        self.W = self.b = None

    def build(self, n_in, rng):
        self.W = init_weights(self.initializer, n_in, self.units, rng)
        self.b = np.zeros(self.units)
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        self.z = x @ self.W + self.b
        self.a = ACTIVATIONS[self.activation][0](self.z)
        return self.a

    def backward(self, grad_a):
        grad_z = ACTIVATIONS[self.activation][1](self.z, self.a, grad_a)
        self.dW = self.x.T @ grad_z
        self.db = grad_z.sum(axis=0)
        return grad_z @ self.W.T

    def config(self):
        return {"units": self.units, "activation": self.activation,
                "initializer": self.initializer}


# --------------------------------------------------------------------- сеть


class MLP:
    def __init__(self, input_shape, layers, loss="binaryCrossentropy",
                 seed=None):
        if loss not in LOSSES:
            raise ValueError("неизвестная функция потерь: " + loss)
        self.input_shape = input_shape
        self.layers = layers
        self.loss = loss
        rng = np.random.default_rng(seed)
        n_in = input_shape
        for layer in layers:
            layer.build(n_in, rng)
            n_in = layer.units

    # прямое распространение
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict_proba(self, x):
        return self.forward(x)

    # обратное распространение
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, lr, momentum=0.0):
        for l in self.layers:
            l.vW = momentum * l.vW - lr * l.dW
            l.vb = momentum * l.vb - lr * l.db
            l.W += l.vW
            l.b += l.vb

    def evaluate(self, x, y):
        p = self.forward(x)
        loss = LOSSES[self.loss][0](p, y)
        acc = float(np.mean(p.argmax(axis=1) == y.argmax(axis=1)))
        return loss, acc

    def get_weights(self):
        return [(l.W.copy(), l.b.copy()) for l in self.layers]

    def set_weights(self, weights):
        for l, (W, b) in zip(self.layers, weights):
            l.W, l.b = W.copy(), b.copy()

    def fit(self, x_train, y_train, x_valid=None, y_valid=None,
            learning_rate=0.1, batch_size=8, epochs=100, momentum=0.0,
            patience=0, seed=None, verbose=True):
        """Мини-батчевый градиентный спуск. Возвращает историю обучения.

        patience > 0 включает раннюю остановку по val_loss с возвратом
        лучших весов.
        """
        rng = np.random.default_rng(seed)
        loss_grad = LOSSES[self.loss][1]
        n = x_train.shape[0]
        hist = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
        best, best_w, wait = np.inf, None, 0
        width = len(str(epochs))
        for epoch in range(1, epochs + 1):
            idx = rng.permutation(n)
            for s in range(0, n, batch_size):
                b = idx[s:s + batch_size]
                pred = self.forward(x_train[b])
                self.backward(loss_grad(pred, y_train[b]))
                self.update(learning_rate, momentum)
            loss, acc = self.evaluate(x_train, y_train)
            hist["loss"].append(loss)
            hist["acc"].append(acc)
            msg = "epoch %0*d/%d - loss: %.4f - acc: %.4f" % (
                width, epoch, epochs, loss, acc)
            if x_valid is not None:
                vl, va = self.evaluate(x_valid, y_valid)
                hist["val_loss"].append(vl)
                hist["val_acc"].append(va)
                msg += " - val_loss: %.4f - val_acc: %.4f" % (vl, va)
                if patience > 0:
                    if vl < best - 1e-6:
                        best, best_w, wait = vl, self.get_weights(), 0
                    else:
                        wait += 1
                        if wait >= patience:
                            if verbose:
                                print(msg)
                                print("early stopping: лучшая эпоха %d"
                                      % (epoch - wait))
                            self.set_weights(best_w)
                            return hist
            if verbose:
                print(msg)
        if best_w is not None:
            self.set_weights(best_w)
        return hist

    # сохранение/загрузка (без pickle: массивы + JSON-описание архитектуры)
    def save(self, path, extra=None):
        meta = {"input_shape": self.input_shape, "loss": self.loss,
                "layers": [l.config() for l in self.layers],
                "extra": extra or {}}
        arrays = {"meta": np.array(json.dumps(meta))}
        for i, l in enumerate(self.layers):
            arrays["W%d" % i] = l.W
            arrays["b%d" % i] = l.b
        with open(path, "wb") as f:  # файл-объект: расширение не меняется
            np.savez(f, **arrays)

    @staticmethod
    def load(path):
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        layers = [DenseLayer(c["units"], c["activation"], c["initializer"])
                  for c in meta["layers"]]
        net = MLP(meta["input_shape"], layers, meta["loss"])
        for i, l in enumerate(net.layers):
            l.W, l.b = data["W%d" % i], data["b%d" % i]
        return net, meta["extra"]


# ------------------------------------------------------------------- данные

CLASSES = ["B", "M"]  # индекс 0 - доброкачественная, 1 - злокачественная


def load_csv(path):
    """Читает CSV без заголовка: id, диагноз (M/B), 30 признаков."""
    import pandas as pd
    df = pd.read_csv(path, header=None)
    if str(df.iloc[0, 1]) not in CLASSES:  # файл с заголовком
        df = pd.read_csv(path)
        df.columns = range(df.shape[1])
    return df


def split_xy(df):
    x = df.iloc[:, 2:].to_numpy(dtype=float)
    labels = df.iloc[:, 1].astype(str).to_numpy()
    y = np.array([CLASSES.index(v) for v in labels])
    return x, y


def one_hot(y, k=2):
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1.0
    return out


def standardize_fit(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


# ------------------------------------------------------------ обучение


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", help="JSON-файл с параметрами (ключи как у аргументов)")
    p.add_argument("--train", default="data_training.csv")
    p.add_argument("--valid", default="data_valid.csv")
    p.add_argument("--layer", type=int, nargs="+", default=[24, 24, 24],
                   help="размеры скрытых слоёв")
    p.add_argument("--activation", default="sigmoid",
                   choices=["sigmoid", "relu", "tanh"],
                   help="активация скрытых слоёв")
    p.add_argument("--initializer", default="heUniform",
                   choices=["heUniform", "heNormal", "xavierUniform",
                            "xavierNormal", "normal"])
    p.add_argument("--loss", default="binaryCrossentropy",
                   choices=["binaryCrossentropy", "categoricalCrossentropy"])
    p.add_argument("--epochs", type=int, default=84)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=0.0314)
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=0,
                   help="ранняя остановка по val_loss (0 - выключена)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", default="saved_model.npy")
    p.add_argument("--plot", default="learning_curves.png")
    p.add_argument("--show", action="store_true", help="показать графики")
    p.add_argument("--gradcheck", action="store_true",
                   help="проверить градиенты численно и выйти")
    a = p.parse_args()
    if a.config:
        with open(a.config, encoding="utf-8") as f:
            cfg = json.load(f)
        p.set_defaults(**cfg)
        a = p.parse_args()
    return a


def plot_curves(hist, path, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = np.arange(1, len(hist["loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, hist["loss"], label="training loss")
    ax[0].plot(ep, hist["val_loss"], "--", label="validation loss")
    ax[0].set(xlabel="Epochs", ylabel="Loss", title="Learning Curves: loss")
    ax[1].plot(ep, hist["acc"], label="training acc")
    ax[1].plot(ep, hist["val_acc"], "--", label="validation acc")
    ax[1].set(xlabel="Epochs", ylabel="Accuracy",
              title="Learning Curves: accuracy")
    for x in ax:
        x.grid(alpha=0.3)
        x.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()


def gradient_check():
    """Сравнение градиентов обратного распространения с численными."""
    for loss in LOSSES:
        for act in ("sigmoid", "relu", "tanh"):
            rng = np.random.default_rng(0)
            x = rng.normal(size=(6, 5))
            y = one_hot(rng.integers(0, 2, 6))
            net = MLP(5, [DenseLayer(4, act), DenseLayer(3, act),
                          DenseLayer(2, "softmax")], loss, seed=1)
            for l in net.layers:  # z=0 - точка излома ReLU, берём b != 0
                l.b = rng.normal(0.0, 0.5, l.b.shape)
            f, g = LOSSES[loss]
            net.backward(g(net.forward(x), y))
            worst, h = 0.0, 1e-6
            for l in net.layers:
                for P, dP in ((l.W, l.dW), (l.b, l.db)):
                    for i in np.ndindex(P.shape):
                        old = P[i]
                        P[i] = old + h
                        up = f(net.forward(x), y)
                        P[i] = old - h
                        down = f(net.forward(x), y)
                        P[i] = old
                        num = (up - down) / (2 * h)
                        worst = max(worst, abs(num - dP[i]) /
                                    max(1e-8, abs(num) + abs(dP[i])))
            print("%-24s %-8s max rel. error: %.2e" % (loss, act, worst))


def main():
    a = parse_args()
    if a.gradcheck:
        gradient_check()
        return
    x_tr, y_tr = split_xy(load_csv(a.train))
    x_va, y_va = split_xy(load_csv(a.valid))
    print("x_train shape : %s" % (x_tr.shape,))
    print("x_valid shape : %s" % (x_va.shape,))

    # масштабирование: статистики считаются только по обучающей части
    mean, std = standardize_fit(x_tr)
    x_tr, x_va = (x_tr - mean) / std, (x_va - mean) / std
    y_tr, y_va = one_hot(y_tr), one_hot(y_va)

    layers = [DenseLayer(u, a.activation, a.initializer) for u in a.layer]
    layers.append(DenseLayer(2, "softmax", a.initializer))
    net = MLP(x_tr.shape[1], layers, a.loss, seed=a.seed)
    hist = net.fit(x_tr, y_tr, x_va, y_va, learning_rate=a.learning_rate,
                   batch_size=a.batch_size, epochs=a.epochs,
                   momentum=a.momentum, patience=a.patience, seed=a.seed)

    net.save(a.save, extra={"mean": mean.tolist(), "std": std.tolist(),
                            "classes": ["B", "M"]})
    print("> saving model '%s' to disk..." % a.save)
    plot_curves(hist, a.plot, a.show)
    print("> learning curves saved to '%s'" % a.plot)


if __name__ == "__main__":
    main()
