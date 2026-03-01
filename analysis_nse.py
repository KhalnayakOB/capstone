# analysis_nse.py  -- corrected to handle multi-dimensional Y safely
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

npz_path = "C:/Users/bhava/OneDrive/Desktop/capstone/data/processed/expert_data_rrt_3d_v2_1000_noisy.npz"

# --------- LOAD NPZ ----------
data = np.load(npz_path)
print("NPZ keys:", data.files)

X = data['X']   # features
Y = data['Y']   # targets (could be multidimensional)
print("X.shape:", X.shape)
print("Y.shape:", Y.shape)

# ---------- DECISION: which target to use? ----------
# By default we pick target_idx = 0 (first column). Change this if you want column 1 or 2.
target_idx = 0

# If Y is 1D, keep as is. If Y is 2D, pick the chosen column.
if Y.ndim == 1:
    Y1 = Y.reshape(-1)
else:
    # Validate index
    if target_idx < 0 or target_idx >= Y.shape[1]:
        raise ValueError(f"target_idx {target_idx} out of range for Y with shape {Y.shape}")
    Y1 = Y[:, target_idx]

print(f"Using Y[:, {target_idx}] as the target. Preview:")
print(Y1[:5])

# If you want to compute NSE for each Y column later, see the block near the end.

# ---------- BUILD DataFrame ----------
if X.ndim == 1 or X.shape[1] == 1:
    feat_cols = ['f0']
    X2 = X.reshape(-1,1)
else:
    feat_cols = [f"f{i}" for i in range(X.shape[1])]
    X2 = X

df = pd.DataFrame(X2, columns=feat_cols)
df['target'] = Y1
print("\nDataFrame head:")
print(df.head())
n = len(df)
print("Total rows:", n)

# ---------- TIME-ORDER ASSUMPTION & SPLIT ----------
# (If your data is NOT time-series change val_frac and splitting method)
val_frac = 0.15
val_start = int(n * (1 - val_frac))
train_df = df.iloc[:val_start].reset_index(drop=True)
val_df   = df.iloc[val_start:].reset_index(drop=True)
print("Train rows:", len(train_df), "Val rows:", len(val_df))

# ---------- NSE function ----------
def nse_score(y_true, y_pred):
    num = np.sum((y_true - y_pred)**2)
    den = np.sum((y_true - np.mean(y_true))**2)
    return 1 - num/den

# ---------- BASELINE (mean) ----------
y_val = val_df['target'].values
y_mean_pred = np.repeat(train_df['target'].mean(), len(y_val))
baseline_nse = nse_score(y_val, y_mean_pred)
print("\nBaseline (mean) NSE:", baseline_nse)

# ---------- ESTIMATE MAX NSE (noise ceiling) ----------
y_all = df['target'].values
print("\nEstimated max NSE using smoothing (several windows):")
for w in [3,5,7,12,24]:
    y_smooth = pd.Series(y_all).rolling(window=w, min_periods=1, center=True).mean().values
    noise_var = np.var(y_all - y_smooth, ddof=0)
    total_var = np.var(y_all, ddof=0)
    max_nse_est = 1 - noise_var / total_var
    print(f"  window={w:2d}: max_NSE ≈ {max_nse_est:.4f}")

# ---------- PERSISTENCE ----------
combined = pd.concat([train_df, val_df], ignore_index=True)
persist_pred = combined['target'].shift(1).fillna(method='bfill')
start_idx = len(train_df)
y_persist = persist_pred.iloc[start_idx:start_idx + len(val_df)].values
persist_nse = nse_score(val_df['target'].values, y_persist)
print("\nPersistence NSE (predict previous timestep):", persist_nse)

# ---------- SIMPLE LAG FEATURES ----------
def make_features_on_full_df(df, lags=[1,2,3,24]):
    Xf = df.copy()
    for l in lags:
        Xf[f'lag_{l}'] = Xf['target'].shift(l)
    Xf['roll_mean_3'] = Xf['target'].shift(1).rolling(window=3, min_periods=1).mean()
    Xf['roll_std_3']  = Xf['target'].shift(1).rolling(window=3, min_periods=1).std().fillna(0)
    Xf = Xf.dropna().reset_index(drop=True)
    return Xf

fe_df = make_features_on_full_df(df, lags=[1,2,3,24])
print("\nAfter creating lag features, new length:", len(fe_df))
n_fe = len(fe_df)
val_start_fe = int(n_fe * (1 - val_frac))
train_fe = fe_df.iloc[:val_start_fe].reset_index(drop=True)
val_fe   = fe_df.iloc[val_start_fe:].reset_index(drop=True)
features = [c for c in train_fe.columns if c not in ['target']]
print("Feature columns:", features)
X_train, y_train = train_fe[features], train_fe['target']
X_val, y_val = val_fe[features], val_fe['target']
print("X_train shape:", X_train.shape, "X_val shape:", X_val.shape)

# ---------- LIGHTGBM ----------
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {"objective": "regression", "metric": "l2", "learning_rate": 0.05, "num_leaves": 31, "verbose": -1}
# ---------- LIGHTGBM using sklearn API (more compatible) ----------
from lightgbm import LGBMRegressor

model = LGBMRegressor(objective='regression',
                      learning_rate=0.05,
                      num_leaves=31,
                      n_estimators=500,
                      random_state=42)

# Fit (no verbose/early-stopping to avoid API incompatibilities)
model.fit(X_train, y_train)

# Predict on validation and compute NSE
y_pred = model.predict(X_val)
lgb_nse = nse_score(y_val.values, y_pred)
print("\nLightGBM val NSE (sklearn API):", lgb_nse)


# ---------- RESIDUAL PLOTS ----------
resid = y_val.values - y_pred
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(y_pred, resid, s=8, alpha=0.6); plt.axhline(0, color='k', linewidth=0.7)
plt.xlabel('Predicted'); plt.ylabel('Residual'); plt.title('Residual vs Predicted')
plt.subplot(1,2,2)
sns.histplot(resid, kde=True); plt.title('Residual distribution')
plt.tight_layout()
plt.show()

# ---------- TimeSeries CV ----------
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []
print("\nTimeSeries CV (LightGBM) - fold NSEs:")
for i, (tr_idx, te_idx) in enumerate(tscv.split(fe_df)):
    tr = fe_df.iloc[tr_idx]; te = fe_df.iloc[te_idx]
    X_tr, y_tr = tr[features], tr['target']
    X_te, y_te = te[features], te['target']
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dte = lgb.Dataset(X_te, label=y_te, reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dte], early_stopping_rounds=50, verbose_eval=False)
    y_te_pred = model.predict(X_te, num_iteration=model.best_iteration)
    score = nse_score(y_te.values, y_te_pred)
    cv_scores.append(score)
    print(f"  Fold {i+1}: NSE = {score:.4f}")

print("\n--- SUMMARY ---")
print("Baseline mean NSE:", baseline_nse)
print("Persistence NSE:", persist_nse)
print("LightGBM val NSE:", lgb_nse)
print("TimeSeries CV mean NSE:", np.mean(cv_scores))
print("target_idx used:", target_idx)
