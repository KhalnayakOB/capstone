# hybrid_final.py
# Research-grade DRMME hybrid final pipeline
# - two-stage smoothing + residual
# - LightGBM (optionally Optuna-tuned)
# - Transformer (higher capacity, multi-window ensemble)
# - heteroscedastic residual loss for transformer
# - LSTM baseline (lightweight)
# - stacking (RidgeCV or optional small Meta-MLP)
#
# Run:
#    python hybrid_final.py

import os
import math
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
import joblib
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# Torch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Optional: optuna for LGBM tuning (toggle below)
try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

# ---------------- CONFIG ----------------
NPZ_PATH = r"C:\Users\bhava\OneDrive\Desktop\capstone\data\processed\expert_data_rrt_3d_v2_1000_noisy.npz"
OUT_DIR = Path("hybrid_final_out")
OUT_DIR.mkdir(exist_ok=True)
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Main hyperparameters (tune these)
WINDOWS = [80, 160]               # windows to train transformer on (ensemble)
BATCH = 64
EPOCHS_SMOOTH = 40                # epochs for smooth models
EPOCHS_RESID = 120                # longer for residual models (weak signal)
TRANS_D_MODEL = 256               # transformer width
TRANS_LAYERS = 4
TRANS_HEADS = 8
LR_SMOOTH = 1e-3
LR_RESID = 5e-4

DO_OPTUNA_LGB = False             # set True to run optuna tuning for LightGBM (requires optuna)
LGB_DEFAULT_PARAMS = {
    "n_estimators": 4000,
    "learning_rate": 0.02,
    "num_leaves": 64,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
}

RANDOM_STATE = SEED
VAL_FRAC = 0.15
SMOOTH_WIN = 5                    # smoothing window (experiment: 3,5,9,15)

np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------- utils ----------------
def nse(y, yhat):
    y = np.array(y); yhat = np.array(yhat)
    num = ((y - yhat) ** 2).sum()
    den = ((y - y.mean()) ** 2).sum()
    return 1 - num / den

def rmse(y, yhat):
    return mean_squared_error(y, yhat) ** 0.5

# ---------------- load data ----------------
print("Loading NPZ:", NPZ_PATH)
npz = np.load(NPZ_PATH)
X = npz["X"]
Y = npz["Y"][:, 0]
print("X.shape", X.shape, "Y.shape", Y.shape)

feat_cols = [f"f{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feat_cols)
df["target"] = Y

# ---------------- smoothing + residual ----------------
df["target_smooth"] = df["target"].rolling(SMOOTH_WIN, center=True, min_periods=1).mean()
df["target_resid"] = df["target"] - df["target_smooth"]
print("Smoothing done. target var:", df["target"].var(), "smooth var:", df["target_smooth"].var(), "resid var:", df["target_resid"].var())

# ---------------- feature engineering for LGB ----------------
def build_tab_features(df):
    D = df.copy()
    # target lags
    for lag in [1,2,3,5,10,20,40,80]:
        D[f"t_lag_{lag}"] = D["target"].shift(lag)
    # rolling stats
    for w in [3,5,10,20]:
        D[f"t_rm_{w}"] = D["target"].shift(1).rolling(w, min_periods=1).mean()
        D[f"t_rs_{w}"] = D["target"].shift(1).rolling(w, min_periods=1).std().fillna(0)
    # feature lags
    for f in feat_cols:
        for lag in [1,2,3,5]:
            D[f"{f}_lag_{lag}"] = D[f].shift(lag)
    # diffs
    D["t_diff_1"] = D["target"] - D["target"].shift(1)
    return D.dropna().reset_index(drop=True)

fe = build_tab_features(df)
print("Tab features shape:", fe.shape)
# train/val split (fe used by LGB)
split_idx = int(len(fe) * (1 - VAL_FRAC))
train_df = fe.iloc[:split_idx].reset_index(drop=True)
val_df = fe.iloc[split_idx:].reset_index(drop=True)
tab_features = [c for c in train_df.columns if c not in ["target", "target_smooth", "target_resid"]]

X_tr_tab = train_df[tab_features].values
X_val_tab = val_df[tab_features].values
y_tr_smooth = train_df["target_smooth"].values
y_val_smooth = val_df["target_smooth"].values
y_tr_res = train_df["target_resid"].values
y_val_res = val_df["target_resid"].values

# ---------------- LightGBM (optionally Optuna tune) ----------------
def run_optuna_lgb(Xtr, ytr, Xv, yv, n_trials=40):
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna not available in environment. Set DO_OPTUNA_LGB=False or install optuna.")
    def objective(trial):
        params = {
            'objective': 'regression',
            'num_leaves': trial.suggest_int('num_leaves', 31, 512),
            'learning_rate': trial.suggest_loguniform('lr', 0.005, 0.05),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample', 0.5, 1.0),
            'reg_alpha': trial.suggest_loguniform('alpha', 1e-8, 10.0),
            'reg_lambda': trial.suggest_loguniform('lambda', 1e-8, 10.0),
            'min_child_samples': trial.suggest_int('mcs', 5, 200),
        }
        model = lgb.LGBMRegressor(**params, n_estimators=4000)
        model.fit(Xtr, ytr, eval_set=[(Xv,yv)], early_stopping_rounds=100, verbose=False)
        pred = model.predict(Xv, num_iteration=model.best_iteration_)
        return 1 - ((yv - pred)**2).sum() / ((yv - yv.mean())**2).sum()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print("Optuna best:", study.best_params, "best NSE:", study.best_value)
    return study.best_params

print("Training LightGBM models (smooth + residual). DO_OPTUNA_LGB:", DO_OPTUNA_LGB)
if DO_OPTUNA_LGB:
    if not OPTUNA_AVAILABLE:
        print("Optuna not installed; skipping optuna")
        lgb_params = LGB_DEFAULT_PARAMS
    else:
        best = run_optuna_lgb(X_tr_tab, y_tr_smooth, X_val_tab, y_val_smooth, n_trials=40)
        lgb_params = best
        lgb_params.update({"n_estimators": 4000})
else:
    lgb_params = LGB_DEFAULT_PARAMS

lgb_smooth = lgb.LGBMRegressor(**lgb_params)
try:
    lgb_smooth.fit(X_tr_tab, y_tr_smooth, eval_set=[(X_val_tab, y_val_smooth)], early_stopping_rounds=100, verbose=200)
except Exception:
    lgb_smooth.fit(X_tr_tab, y_tr_smooth)

pred_lgb_smooth = lgb_smooth.predict(X_val_tab)
print("LGB smooth NSE:", nse(y_val_smooth, pred_lgb_smooth), "RMSE:", rmse(y_val_smooth, pred_lgb_smooth))
joblib.dump(lgb_smooth, OUT_DIR/"lgb_smooth.joblib")
np.save(OUT_DIR/"pred_lgb_smooth.npy", pred_lgb_smooth)

# residual LGB uses same tab features but target_resid
lgb_res_params = lgb_params.copy()
lgb_res = lgb.LGBMRegressor(**lgb_res_params)
try:
    lgb_res.fit(X_tr_tab, y_tr_res, eval_set=[(X_val_tab, y_val_res)], early_stopping_rounds=100, verbose=200)
except Exception:
    lgb_res.fit(X_tr_tab, y_tr_res)
pred_lgb_res = lgb_res.predict(X_val_tab)
print("LGB resid NSE:", nse(y_val_res, pred_lgb_res), "RMSE:", rmse(y_val_res, pred_lgb_res))
joblib.dump(lgb_res, OUT_DIR/"lgb_res.joblib")
np.save(OUT_DIR/"pred_lgb_res.npy", pred_lgb_res)

# ---------------- Sequence datasets for transformer/lstm ----------------
# We'll create sequences from the same fe table (so LGB and sequences align with fe indices)
seq_cols = [c for c in fe.columns if c.startswith("f")] + ["target"]
fe_seq = fe[seq_cols]
targets_s = fe["target_smooth"].values
targets_r = fe["target_resid"].values

# build sequences for a largest window if we will train multi-window, we will re-slice accordingly
def build_sequences(arr, targets, window):
    seqs = []
    ys = []
    for i in range(window, len(arr)):
        seqs.append(arr[i-window:i])
        ys.append(targets[i])
    return np.array(seqs), np.array(ys)

arr_all = fe_seq.values  # aligned with fe
# We'll for each window train using sequences built with that window

# ---------------- Torch helper and models ----------------
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class TransformerHetero(nn.Module):
    def __init__(self, input_dim, d_model=TRANS_D_MODEL, nhead=TRANS_HEADS, nlayers=TRANS_LAYERS):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, batch_first=True, dropout=0.2)
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.mean_head = nn.Linear(d_model, 1)
        self.var_head = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos(x)
        x = self.trans(x)
        h = x[:, -1, :]
        mean = self.mean_head(h).squeeze(-1)
        logvar = self.var_head(h).squeeze(-1)
        return mean, logvar

class LSTMReg(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)

# ---------------- Train helper for transformer (heteroscedastic) ----------------
def train_transformer(window, mode="smooth", epochs=EPOCHS_SMOOTH, lr=LR_SMOOTH):
    # build data
    seqs, ys = build_sequences(arr_all, targets_s if mode=="smooth" else targets_r, window)
    # split
    split = int(len(seqs) * (1 - VAL_FRAC))
    Xtr, Xv = seqs[:split], seqs[split:]
    ytr, yv = ys[:split], ys[split:]
    # scale each feature
    n_feats = Xtr.shape[2]
    scalers = []
    for f in range(n_feats):
        s = StandardScaler().fit(Xtr[:,:,f].reshape(-1,1))
        scalers.append(s)
        Xtr[:,:,f] = s.transform(Xtr[:,:,f].reshape(-1,1)).reshape(Xtr[:,:,f].shape)
        Xv[:,:,f] = s.transform(Xv[:,:,f].reshape(-1,1)).reshape(Xv[:,:,f].shape)
    sy = StandardScaler().fit(ytr.reshape(-1,1))
    ytr_s = sy.transform(ytr.reshape(-1,1)).ravel()
    yv_s = sy.transform(yv.reshape(-1,1)).ravel()

    train_dl = DataLoader(SeqDataset(Xtr, ytr_s), batch_size=BATCH, shuffle=True, drop_last=True)
    val_dl = DataLoader(SeqDataset(Xv, yv_s), batch_size=BATCH, shuffle=False)

    model = TransformerHetero(n_feats).to(DEVICE)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=6, verbose=True)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

    best_score = -1e9
    best_path = OUT_DIR / f"transformer_{mode}_w{window}.pth"

    for epoch in range(1, epochs+1):
        model.train()
        train_losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
                mean, logvar = model(xb)
                loss = torch.mean((mean - yb)**2 * torch.exp(-logvar) + logvar)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(opt); scaler_amp.update()
            train_losses.append(loss.item())
        # validation
        model.eval()
        preds = []
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                mean, logvar = model(xb)
                vloss = torch.mean((mean - yb)**2 * torch.exp(-logvar) + logvar)
                val_losses.append(vloss.item())
                preds.append(mean.cpu().numpy())
        preds = np.concatenate(preds)
        preds_orig = sy.inverse_transform(preds.reshape(-1,1)).ravel()
        truths = yv  # original scale target array
        # align lengths
        truths_orig = truths[:len(preds_orig)]
        score = nse(truths_orig, preds_orig)
        print(f"Transformer-{mode} window {window} epoch {epoch} val_nse {score:.4f} val_rmse {rmse(truths_orig,preds_orig):.4f}")
        scheduler.step(np.mean(val_losses))
        if score > best_score + 1e-5:
            best_score = score
            torch.save(model.state_dict(), best_path)
    # load best and predict full val sequence
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in val_dl:
            xb = xb.to(DEVICE)
            mean, _ = model(xb)
            preds.append(mean.cpu().numpy())
    preds = np.concatenate(preds)
    preds_orig = sy.inverse_transform(preds.reshape(-1,1)).ravel()
    np.save(OUT_DIR / f"pred_trans_{mode}_w{window}.npy", preds_orig)
    print(f"Transformer-{mode} window{window} best NSE {best_score:.4f}")
    return preds_orig

# -------------- Train transformer models for each window and both smooth/resid --------------
print("Training Transformer models for windows:", WINDOWS)
all_trans_preds = {}
for w in WINDOWS:
    print("Training transformer smooth window", w)
    preds_s = train_transformer(w, mode="smooth", epochs=EPOCHS_SMOOTH, lr=LR_SMOOTH)
    all_trans_preds[f"trans_s_w{w}"] = preds_s
    print("Training transformer resid window", w)
    preds_r = train_transformer(w, mode="resid", epochs=EPOCHS_RESID, lr=LR_RESID)
    all_trans_preds[f"trans_r_w{w}"] = preds_r

# ---------------- LSTM training (for smooth + resid) - lighter capacity ----------------
def train_lstm(window, mode="smooth", epochs=30, lr=1e-3):
    seqs, ys = build_sequences(arr_all, targets_s if mode=="smooth" else targets_r, window)
    split = int(len(seqs) * (1 - VAL_FRAC))
    Xtr, Xv = seqs[:split], seqs[split:]
    ytr, yv = ys[:split], ys[split:]
    n_feats = Xtr.shape[2]
    scalers_local = []
    for f in range(n_feats):
        s = StandardScaler().fit(Xtr[:,:,f].reshape(-1,1))
        scalers_local.append(s)
        Xtr[:,:,f] = s.transform(Xtr[:,:,f].reshape(-1,1)).reshape(Xtr[:,:,f].shape)
        Xv[:,:,f] = s.transform(Xv[:,:,f].reshape(-1,1)).reshape(Xv[:,:,f].shape)
    sy_local = StandardScaler().fit(ytr.reshape(-1,1))
    ytr_s = sy_local.transform(ytr.reshape(-1,1)).ravel()
    yv_s = sy_local.transform(yv.reshape(-1,1)).ravel()
    train_dl = DataLoader(SeqDataset(Xtr, ytr_s), batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(SeqDataset(Xv, yv_s), batch_size=BATCH, shuffle=False)
    model = LSTMReg(n_feats).to(DEVICE)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    best = -1e9
    best_path = OUT_DIR / f"lstm_{mode}_w{window}.pth"
    for epoch in range(1, epochs+1):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = ((pred - yb)**2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        # val
        model.eval()
        preds=[]
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(DEVICE)
                preds.append(model(xb).cpu().numpy())
        preds = np.concatenate(preds)
        preds_orig = sy_local.inverse_transform(preds.reshape(-1,1)).ravel()
        truths = yv[:len(preds_orig)]
        score = nse(truths, preds_orig)
        print(f"LSTM-{mode} w{window} epoch {epoch} val_nse {score:.4f}")
        if score > best:
            best = score
            torch.save(model.state_dict(), best_path)
    # final preds
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()
    preds=[]
    with torch.no_grad():
        for xb,_ in val_dl:
            xb = xb.to(DEVICE)
            preds.append(model(xb).cpu().numpy())
    preds = np.concatenate(preds)
    preds_orig = sy_local.inverse_transform(preds.reshape(-1,1)).ravel()
    np.save(OUT_DIR / f"pred_lstm_{mode}_w{window}.npy", preds_orig)
    print(f"LSTM-{mode} w{window} best NSE {best:.4f}")
    return preds_orig

all_lstm_preds = {}
for w in WINDOWS:
    print("Training LSTM smooth w", w)
    all_lstm_preds[f"lstm_s_w{w}"] = train_lstm(w, mode="smooth", epochs=40, lr=1e-3)
    print("Training LSTM resid w", w)
    all_lstm_preds[f"lstm_r_w{w}"] = train_lstm(w, mode="resid", epochs=80, lr=5e-4)

# ---------------- Assemble full predictions (smooth + residual) ----------------
# For LGB we have pred_lgb_smooth (length len(val_df)) and pred_lgb_res (length len(val_df))
# For transformers/lstms predictions lengths depend on windows; we will align by trimming to min length.

# Compose predicted smooth+resid for each model type and window
composed_preds = {}

# LGB full (already aligned with val_df)
pred_lgb_full = pred_lgb_smooth + pred_lgb_res  # both are length len(val_df)
composed_preds["lgb_full"] = pred_lgb_full

# Transformer full for each window
for w in WINDOWS:
    key_s = f"trans_s_w{w}"
    key_r = f"trans_r_w{w}"
    p_s = all_trans_preds[key_s]
    p_r = all_trans_preds[key_r]
    # ensure lengths match by taking last min length
    m = min(len(p_s), len(p_r))
    composed_preds[f"trans_full_w{w}"] = p_s[-m:] + p_r[-m:]

# LSTM full for each window
for w in WINDOWS:
    p_s = all_lstm_preds[f"lstm_s_w{w}"]
    p_r = all_lstm_preds[f"lstm_r_w{w}"]
    m = min(len(p_s), len(p_r))
    composed_preds[f"lstm_full_w{w}"] = p_s[-m:] + p_r[-m:]

# ---------------- Build stack training arrays (align by trimming to common min length) ----------------
# The ground truth for stacking should be val_df["target"], but length may differ. We'll align to last min length across preds.
all_lengths = [len(arr) for arr in composed_preds.values()]
min_len = min(all_lengths)
print("Prediction lengths:", {k: len(v) for k,v in composed_preds.items()}, "min_len:", min_len)

# Collect arrays trimmed to last min_len
stack_X = np.vstack([v[-min_len:] for v in composed_preds.values()]).T
truth_full = val_df["target"].values[-min_len:]

# ---------------- Stacking meta-learner ----------------
meta = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3).fit(stack_X, truth_full)
final_pred = meta.predict(stack_X)
print("Stacking coefs:", meta.coef_)
print("STACKED HYBRID NSE:", nse(truth_full, final_pred), "RMSE:", rmse(truth_full, final_pred))
np.save(OUT_DIR/"hybrid_final_pred.npy", final_pred)
joblib.dump(meta, OUT_DIR/"hybrid_meta.joblib")

# Optionally print per-model NSEs (trimmed)
print("\nPer-model NSE (trimmed to stack length):")
for name, arr in composed_preds.items():
    arr_t = arr[-min_len:]
    print(f"{name} NSE:", nse(truth_full, arr_t), "RMSE:", rmse(truth_full, arr_t))

print("\nSaved outputs to:", OUT_DIR)
