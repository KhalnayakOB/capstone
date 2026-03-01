# ensemble_stack.py
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# assume you saved predictions from previous runs as npy:
# lgb_preds_val.npy, nn_preds_val.npy, lstm_preds_val.npy  (these are validation-set predictions)
y_val = np.load("val_true.npy")   # ground truth for validation window used in those models
a = np.load("lgb_preds_val.npy")
b = np.load("nn_preds_val.npy")
c = np.load("lstm_preds_val.npy")

Xstack = np.vstack([a,b,c]).T
meta = LinearRegression().fit(Xstack, y_val)
stack_pred = meta.predict(Xstack)
print("Stack NSE:", 1 - ((y_val-stack_pred)**2).sum() / ((y_val-y_val.mean())**2).sum())
print("Stack RMSE:", mean_squared_error(y_val, stack_pred, squared=False))
