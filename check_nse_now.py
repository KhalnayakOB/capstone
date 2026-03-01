import numpy as np

def safe_nse(y_true, y_pred):
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)


# Ground truth
y_true = np.load("val_truth.npy")

models = {
    "LightGBM": "lgb_preds.npy",
    "MLP": "mlp_preds.npy",
    "LSTM": "lstm_preds.npy",
}

for name, file in models.items():
    try:
        y_pred = np.load(file)
        print(f"{name} NSE = {safe_nse(y_true, y_pred):.4f}")

    except FileNotFoundError:
        print(f"{name}: prediction file not found")
