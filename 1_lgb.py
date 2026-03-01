import os
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

def nse(y, yhat):
    return 1 - ((y - yhat)**2).sum() / ((y - y.mean())**2).sum()

# ---------------- CONFIG ----------------
NPZ_PATH = r"C:\Users\bhava\OneDrive\Desktop\capstone\data\processed\expert_data_rrt_3d_v2_1000_noisy.npz"
TARGET_IDX = 0

print("Loading:", NPZ_PATH)
npz = np.load(NPZ_PATH)
X = npz['X']
Y = npz['Y'][:, TARGET_IDX]

df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
df['target'] = Y

def add_features(df):
    d = df.copy()
    for l in [1,2,3,5,10,20,40,80]:
        d[f"lag_{l}"] = d["target"].shift(l)
    for w in [3,5,10,20]:
        d[f"roll_mean_{w}"] = d["target"].shift(1).rolling(w).mean()
        d[f"roll_std_{w}"] = d["target"].shift(1).rolling(w).std()
    return d.dropna().reset_index(drop=True)

fe = add_features(df)

split = int(len(fe)*0.85)
train = fe.iloc[:split]
val   = fe.iloc[split:]

features = [c for c in fe.columns if c != "target"]
Xtr, ytr = train[features], train["target"]
Xv, yv   = val[features], val["target"]

model = LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8
)

model.fit(Xtr, ytr)

pred = model.predict(Xv)
lgb_nse = nse(yv.values, pred)
lgb_rmse = mean_squared_error(yv.values, pred)**0.5

print("\nLGBM NSE:", lgb_nse)
print("LGBM RMSE:", lgb_rmse)

np.save("lgb_preds.npy", pred)
np.save("val_truth.npy", yv.values)
print("Saved: lgb_preds.npy, val_truth.npy")
