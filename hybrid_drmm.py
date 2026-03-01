"""
hybrid_drmm.py
Full hybrid DRMME pipeline:
 - Target smoothing
 - Residual modeling
 - LightGBM (smooth + residual)
 - Transformer (smooth + residual, heteroscedastic)
 - LSTM (smooth + residual)
 - Stacking meta-learner
"""

import numpy as np
import pandas as pd
import os, json, math
from pathlib import Path

# ML libraries
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

# Torch libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ======================================================
# Configuration
# ======================================================

NPZ_PATH = r"C:\Users\bhava\OneDrive\Desktop\capstone\data\processed\expert_data_rrt_3d_v2_1000_noisy.npz"
OUT = Path("hybrid_out")
OUT.mkdir(exist_ok=True)

WINDOW = 80
BATCH = 64
EPOCHS = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# ======================================================
# Utility functions
# ======================================================

def nse(y, yhat):
    return 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()

def rmse(y, yhat):
    return mean_squared_error(y, yhat) ** 0.5


# ======================================================
# Load Data
# ======================================================

npz = np.load(NPZ_PATH)
X = npz["X"]
Y = npz["Y"][:, 0]

df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
df["target"] = Y

print("Loaded:", df.shape)


# ======================================================
# Stage 1 — Target Smoothing (Trend Extraction)
# ======================================================

SMOOTH_WIN = 5
df["target_smooth"] = df["target"].rolling(SMOOTH_WIN, center=True, min_periods=1).mean()
df["target_resid"] = df["target"] - df["target_smooth"]

print("Smooth variance:", df["target_smooth"].var(), 
      "Resid variance:", df["target_resid"].var())


# ======================================================
# Feature Engineering for LightGBM
# ======================================================

def build_features(df):
    D = df.copy()
    for lag in [1,2,3,5,10,20,40,80]:
        D[f"t_lag_{lag}"] = D["target"].shift(lag)
    for w in [3,5,10,20]:
        D[f"t_rm_{w}"] = D["target"].shift(1).rolling(w).mean()
        D[f"t_rs_{w}"] = D["target"].shift(1).rolling(w).std().fillna(0)
    for f in [c for c in df.columns if c.startswith("f")]:
        for lag in [1,2,3,5]:
            D[f"{f}_lag_{lag}"] = D[f].shift(lag)
    return D.dropna().reset_index(drop=True)

fe = build_features(df)


# ======================================================
# Train/Val Split
# ======================================================

VAL_FRAC = 0.15
split = int(len(fe) * (1 - VAL_FRAC))

train_df = fe.iloc[:split].reset_index(drop=True)
val_df   = fe.iloc[split:].reset_index(drop=True)

tab_features = [c for c in train_df.columns if c not in ["target", "target_smooth", "target_resid"]]

y_tr_smooth = train_df["target_smooth"].values
y_tr_res    = train_df["target_resid"].values

y_val_smooth = val_df["target_smooth"].values
y_val_res    = val_df["target_resid"].values

X_tr = train_df[tab_features].values
X_val = val_df[tab_features].values


# ======================================================
# Stage 2 — LightGBM (Smooth + Residual)
# ======================================================

def train_lgb(Xtr, ytr, Xv, yv, name):
    model = lgb.LGBMRegressor(
        n_estimators=2500,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        num_leaves=64,
    )
    try:
        model.fit(Xtr, ytr, eval_set=[(Xv, yv)], early_stopping_rounds=100, verbose=100)
    except:
        model.fit(Xtr, ytr)

    pred = model.predict(Xv)
    np.save(OUT / f"pred_lgb_{name}.npy", pred)

    print(f"LGB-{name}: NSE={nse(yv, pred):.4f}, RMSE={rmse(yv, pred):.4f}")
    return pred

pred_lgb_smooth = train_lgb(X_tr, y_tr_smooth, X_val, y_val_smooth, "smooth")
pred_lgb_res    = train_lgb(X_tr, y_tr_res,    X_val, y_val_res,    "resid")


# ======================================================
# Stage 3 — Sequence Data for Transformer & LSTM
# ======================================================

seq_cols = [c for c in df.columns if c.startswith("f")] + ["target"]
fe_seq = fe[seq_cols]

arr = fe_seq.values
targets_s = fe["target_smooth"].values
targets_r = fe["target_resid"].values

seqs, ys_s, ys_r = [], [], []

for i in range(WINDOW, len(arr)):
    seqs.append(arr[i-WINDOW:i])
    ys_s.append(targets_s[i])
    ys_r.append(targets_r[i])

seqs = np.array(seqs)
ys_s = np.array(ys_s)
ys_r = np.array(ys_r)

split2 = int(len(seqs)*(1-VAL_FRAC))

Xtr_seq, Xval_seq = seqs[:split2], seqs[split2:]
ytr_s_seq, yval_s_seq = ys_s[:split2], ys_s[split2:]
ytr_r_seq, yval_r_seq = ys_r[:split2], ys_r[split2:]

# Scale features
n_feats = Xtr_seq.shape[2]
scalers = []
for f in range(n_feats):
    s = StandardScaler().fit(Xtr_seq[:,:,f].reshape(-1,1))
    scalers.append(s)
    Xtr_seq[:,:,f] = s.transform(Xtr_seq[:,:,f].reshape(-1,1)).reshape(Xtr_seq[:,:,f].shape)
    Xval_seq[:,:,f] = s.transform(Xval_seq[:,:,f].reshape(-1,1)).reshape(Xval_seq[:,:,f].shape)

sy_s = StandardScaler().fit(ytr_s_seq.reshape(-1,1))
sy_r = StandardScaler().fit(ytr_r_seq.reshape(-1,1))

ytr_s_norm = sy_s.transform(ytr_s_seq.reshape(-1,1)).ravel()
yval_s_norm = sy_s.transform(yval_s_seq.reshape(-1,1)).ravel()

ytr_r_norm = sy_r.transform(ytr_r_seq.reshape(-1,1)).ravel()
yval_r_norm = sy_r.transform(yval_r_seq.reshape(-1,1)).ravel()


# ======================================================
# Torch Dataset
# ======================================================

class SeqDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader_s = DataLoader(SeqDS(Xtr_seq,ytr_s_norm), batch_size=BATCH, shuffle=True)
val_loader_s   = DataLoader(SeqDS(Xval_seq,yval_s_norm), batch_size=BATCH)

train_loader_r = DataLoader(SeqDS(Xtr_seq,ytr_r_norm), batch_size=BATCH, shuffle=True)
val_loader_r   = DataLoader(SeqDS(Xval_seq,yval_r_norm), batch_size=BATCH)


# ======================================================
# Transformer (with heteroscedastic output)
# ======================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        pe = torch.zeros(1000, d_model)
        pos = torch.arange(0,1000).unsqueeze(1)
        div = torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.pe = pe.unsqueeze(0)
    def forward(self,x):
        return x + self.pe[:,:x.size(1),:].to(x.device)

class TransformerHetero(nn.Module):
    def __init__(self, input_dim, d_model=128, layers=3):
        super().__init__()
        self.inp = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, 8, 256, batch_first=True)
        self.trans = nn.TransformerEncoder(enc_layer, layers)
        self.mean_head = nn.Linear(d_model, 1)
        self.var_head = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.inp(x)
        x = self.pos(x)
        x = self.trans(x)
        h = x[:,-1,:]
        mean = self.mean_head(h)
        logvar = self.var_head(h)
        return mean.squeeze(), logvar.squeeze()


def train_transformer(loader, vloader, yscale, mode_name):
    model = TransformerHetero(n_feats).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = -1e9

    for epoch in range(1,EPOCHS+1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            mean, logvar = model(xb)

            loss = torch.mean((mean - yb)**2 * torch.exp(-logvar) + logvar)
            loss.backward()
            opt.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for xb, yb in vloader:
                xb = xb.to(DEVICE)
                mean, logvar = model(xb)
                preds.append(mean.cpu().numpy())
        preds = np.concatenate(preds)
        preds_orig = yscale.inverse_transform(preds.reshape(-1,1)).ravel()
        truths = yscale.inverse_transform(
                    np.concatenate([yb.numpy() for _,yb in vloader]).reshape(-1,1)
                 ).ravel()
        score = nse(truths, preds_orig)
        print(f"Transformer-{mode_name} epoch {epoch} NSE={score:.4f}")

        if score > best:
            best = score
            torch.save(model.state_dict(), OUT/f"trans_best_{mode_name}.pth")

    print(f"Transformer-{mode_name} best NSE={best:.4f}")

    # final predictions
    model.load_state_dict(torch.load(OUT/f"trans_best_{mode_name}.pth"))
    model.eval()
    preds = []
    with torch.no_grad():
        for xb,_ in vloader:
            xb = xb.to(DEVICE)
            mean, _ = model(xb)
            preds.append(mean.cpu().numpy())
    preds = np.concatenate(preds)
    preds_orig = yscale.inverse_transform(preds.reshape(-1,1)).ravel()
    np.save(OUT/f"pred_trans_{mode_name}.npy", preds_orig)
    return preds_orig


# Train transformer smooth + residual
pred_trans_s = train_transformer(train_loader_s, val_loader_s, sy_s, "smooth")
pred_trans_r = train_transformer(train_loader_r, val_loader_r, sy_r, "resid")


# ======================================================
# LSTM Model
# ======================================================

class LSTMReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(n_feats,128,2,batch_first=True,dropout=0.2)
        self.fc = nn.Linear(128,1)
    def forward(self,x):
        o,_ = self.lstm(x)
        return self.fc(o[:,-1,:]).squeeze()

def train_lstm(loader, vloader, yscale, mode_name):
    model = LSTMReg().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = -1e9

    for epoch in range(1,EPOCHS+1):
        model.train()
        for xb,yb in loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = ((pred - yb)**2).mean()
            loss.backward()
            opt.step()

        model.eval()
        preds=[]
        with torch.no_grad():
            for xb,yb in vloader:
                xb = xb.to(DEVICE)
                preds.append(model(xb).cpu().numpy())
        preds = np.concatenate(preds)
        preds_orig = yscale.inverse_transform(preds.reshape(-1,1)).ravel()
        truths = yscale.inverse_transform(
                    np.concatenate([yb.numpy() for _,yb in vloader]).reshape(-1,1)
                 ).ravel()
        score = nse(truths, preds_orig)
        print(f"LSTM-{mode_name} epoch {epoch}: NSE={score:.4f}")

        if score > best:
            best = score
            torch.save(model.state_dict(), OUT/f"lstm_best_{mode_name}.pth")

    print(f"LSTM-{mode_name} best NSE={best:.4f}")

    # final predictions
    model.load_state_dict(torch.load(OUT/f"lstm_best_{mode_name}.pth"))
    model.eval()
    preds=[]
    with torch.no_grad():
        for xb,_ in vloader:
            xb = xb.to(DEVICE)
            preds.append(model(xb).cpu().numpy())
    preds = np.concatenate(preds)
    preds_orig = yscale.inverse_transform(preds.reshape(-1,1)).ravel()
    np.save(OUT/f"pred_lstm_{mode_name}.npy", preds_orig)
    return preds_orig


# Train LSTM smooth + residual
pred_lstm_s = train_lstm(train_loader_s, val_loader_s, sy_s, "smooth")
pred_lstm_r = train_lstm(train_loader_r, val_loader_r, sy_r, "resid")


# ======================================================
# Combine: Smooth + Residual
# ======================================================

# LightGBM full
pred_lgb_full = (
    pred_lgb_smooth[-len(pred_lgb_res):] + pred_lgb_res
)

# Transformer full
pred_trans_full = (
    pred_trans_s[-len(pred_trans_r):] + pred_trans_r
)

# LSTM full
pred_lstm_full = (
    pred_lstm_s[-len(pred_lstm_r):] + pred_lstm_r
)


# ======================================================
# Final stacking meta-learner
# ======================================================

min_len = min(len(pred_lgb_full), len(pred_trans_full), len(pred_lstm_full))
pred_lgb_full   = pred_lgb_full[-min_len:]
pred_trans_full = pred_trans_full[-min_len:]
pred_lstm_full  = pred_lstm_full[-min_len:]

truth = val_df["target"].values[-min_len:]

StackX = np.vstack([pred_lgb_full, pred_trans_full, pred_lstm_full]).T

meta = RidgeCV(alphas=[0.1,1.0,10.0]).fit(StackX, truth)
final = meta.predict(StackX)

print("Stacking Coefs:", meta.coef_)
print("HYBRID FINAL NSE:", nse(truth, final))
print("HYBRID FINAL RMSE:", rmse(truth, final))

np.save(OUT/"hybrid_final.npy", final)

print("\nHybrid model complete! Outputs in:", OUT)
