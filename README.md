# Многослойный персептрон
## Запуск

Откройте терминал в папке проекта, где находятся три программы и `data.csv`.

Один раз установите зависимости:

```bash
py -m pip install numpy matplotlib
```

Выполните команды по порядку:

```bash
py split_data.py
py train_mlp.py
py predict.py
```

## Программы

- `split_data.py` — анализирует `data.csv`, сохраняет график `data_analysis.png` и делит данные на `data_training.csv` и `data_validation.csv` в пропорции примерно 80/20 с `seed=42`.
- `train_mlp.py` — обучает сеть `30 → 24 → 24 → 2` в течение 120 эпох, выводит метрики и сохраняет модель `saved_model.npz` и графики `learning_curves.png`.
- `predict.py` — загружает модель, делает прогноз для `data_validation.csv`, выводит бинарную кросс-энтропию, точность и матрицу ошибок. Прогнозы сохраняются в `predictions.csv`.

