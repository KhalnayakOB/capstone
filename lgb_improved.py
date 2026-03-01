# lgb_improved.py
# Robust LightGBM baseline with expanded lag/rolling features
# Copy this file exactly and run: python lgb_improved.py

import os
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

# ---------------- utility: NSE ----------------
def nse(y, yhat):
    num = ((y - yhat) ** 2).sum()
    den = ((y - y.mean()) ** 2).sum()
    return 1 - num / den

# ---------------- config ----------------
NPZ_PATH = r"C:\Users\bhava\OneDrive\Desktop\capstone\data\processed\expert_data_rrt_3d_v2_1000_noisy.npz"
TARGET_COL_IDX = 0   # choose which Y column to use (0,1,2)

# ---------------- load with diagnostics ----------------
print("Loading NPZ from:", NPZ_PATH)
print("Exists?", os.path.exists(NPZ_PATH))
if not os.path.exists(NPZ_PATH):
    raise SystemExit(f"ERROR: NPZ file not found at {NPZ_PATH}. Fix the path and re-run.")

npz = np.load(NPZ_PATH)
print("NPZ keys:", npz.files)

# assume convention X -> features, Y -> targets
if 'X' not in npz.files or 'Y' not in npz.files:
    raise SystemExit("ERROR: NPZ does not contain expected keys 'X' and 'Y'.")

X = npz['X']
Y = npz['Y']

print("X.shape:", X.shape)
print("Y.shape:", Y.shape)

# ---------------- choose target ----------------
if Y.ndim == 1:
    y_all = Y.reshape(-1)
else:
    if TARGET_COL_IDX < 0 or TARGET_COL_IDX >= Y.shape[1]:
        raise SystemExit(f"ERROR: TARGET_COL_IDX {TARGET_COL_IDX} out of range (Y has shape {Y.shape})")
    y_all = Y[:, TARGET_COL_IDX]

# ---------------- build DataFrame ----------------
if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
    feat_cols = ['f0']
    X2 = X.reshape(-1, 1)
else:
    feat_cols = [f"f{i}" for i in range(X.shape[1])]
    X2 = X

df = pd.DataFrame(X2, columns=feat_cols)
df['target'] = y_all
print("\nDataFrame preview:")
print(df.head())
print("Total rows:", len(df))

# ---------------- create expanded lag + rolling features ----------------
def add_lags_rolls(df, lags=[1,2,3,5,10,20,40,80], roll_windows=[3,5,10,20]):
    d = df.copy()
    # create lags of the target (these are most important)
    for l in lags:
        d[f'lag_{l}'] = d['target'].shift(l)
    # rolling means and stds of past values (shift by 1 to avoid leakage)
    for w in roll_windows:
        d[f'roll_mean_{w}'] = d['target'].shift(1).rolling(window=w, min_periods=1).mean()
        d[f'roll_std_{w}']  = d['target'].shift(1).rolling(window=w, min_periods=1).std().fillna(0)
    # optionally you can add rolling of features too, but start with target-based lags
    d = d.dropna().reset_index(drop=True)
    return d

print("\nBuilding lag + rolling features (this may drop first rows)...")
fe = add_lags_rolls(df)
print("After feature creation: rows =", len(fe))
val_frac = 0.15
split_idx = int(len(fe) * (1 - val_frac))

train = fe.iloc[:split_idx].reset_index(drop=True)
val   = fe.iloc[split_idx:].reset_index(drop=True)

features = [c for c in train.columns if c != 'target']
Xtr, ytr = train[features], train['target']
Xv,  yv  = val[features], val['target']

print("Train shape:", Xtr.shape, "Val shape:", Xv.shape)
print("Feature count:", len(features))

# ---------------- LightGBM model ----------------
model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Try to use early stopping if supported; otherwise fall back to plain fit
use_early_stop = True
try:
    # sklearn API accepts eval_set and early_stopping_rounds in many installs
    print("\nAttempting to fit with early stopping (if supported)...")
    model.fit(Xtr, ytr, eval_set=[(Xv, yv)], early_stopping_rounds=100, verbose=100)
except TypeError as e:
    print("Early-stopping call failed (TypeError). Falling back to no-early-stopping fit.")
    # fallback: plain fit (safe)
    model = LGBMRegressor(
        n_estimators=500,    # smaller rounds for fallback
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )
    model.fit(Xtr, ytr)

# ---------------- predict & metrics ----------------
# ---------------- predict & metrics ----------------
yhat = model.predict(Xv)
lgb_nse = nse(yv.values, yhat)

# RMSE compatible with all sklearn versions
lgb_rmse = mean_squared_error(yv.values, yhat) ** 0.5

print(f"\nLightGBM val NSE: {lgb_nse:.6f}")
print(f"LightGBM val RMSE: {lgb_rmse:.6f}")


# ---------------- feature importance ----------------
try:
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nTop 30 feature importances:")
    print(importances.head(30))
except Exception as e:
    print("Could not extract feature importances:", e)

# ---------------- save validation predictions & truth for stacking (optional) ----------------
out_dir = os.path.dirname(NPZ_PATH)
np.save(os.path.join(out_dir, "lgb_val_preds.npy"), yhat)
np.save(os.path.join(out_dir, "lgb_val_truth.npy"), yv.values)
print(f"\nSaved validation predictions to: {out_dir}\\lgb_val_preds.npy")
print("Done.")
