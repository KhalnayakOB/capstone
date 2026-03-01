"""
advanced_pipeline.py
Research-grade pipeline: preprocessing, LightGBM, Transformer regressor, LSTM baseline, stacking.
Run: python advanced_pipeline.py
"""

import os, sys, math, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ----------------- Config -----------------
NPZ_PATH = r"C:\Users\bhava\OneDrive\Desktop\capstone\data\processed\expert_data_rrt_3d_v2_1000_noisy.npz"
TARGET_COL_IDX = 0
OUT_DIR = Path("advanced_out")
OUT_DIR.mkdir(exist_ok=True)
SEED = 42
WINDOW = 80            # transformer sequence length (tune: 40,80,160)
BATCH = 64
EPOCHS = 40            # transformer epochs (increase for final runs)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------- Utilities -----------------
def nse(y, yhat):
    num = ((y - yhat)**2).sum()
    den = ((y - y.mean())**2).sum()
    return 1 - num/den

def rmse(y,yhat):
    return mean_squared_error(y,yhat)**0.5

# ----------------- Load & quick inspect -----------------
print("Loading NPZ:", NPZ_PATH)
npz = np.load(NPZ_PATH)
assert 'X' in npz.files and 'Y' in npz.files, f"NPZ missing X/Y keys: {npz.files}"
X = npz['X']
Y = npz['Y'][:, TARGET_COL_IDX]
print("X.shape", X.shape, "Y.shape", Y.shape)

# Build DataFrame (features + target)
feat_cols = [f"f{i}" for i in range(X.shape[1])] if X.ndim==2 else ['f0']
df = pd.DataFrame(X, columns=feat_cols)
df['target'] = Y
n = len(df)

# ----------------- Optional target denoising (experiment) -----------------
# If you want to denoise target use rolling mean. Set DO_DENOISE=True to use.
DO_DENOISE = True
DENoise_WINDOW = 3
if DO_DENOISE:
    df['target_smooth'] = df['target'].rolling(window=DENoise_WINDOW, min_periods=1, center=True).mean()
    # We will predict original target but you can switch to smoothed; below I keep both.
    print(f"Target smoothing applied (window={DENoise_WINDOW}). Var original:", df['target'].var(), "smoothed:", df['target_smooth'].var())

# ----------------- Feature engineering function -----------------
def make_features(df, lags=[1,2,3,5,10,20,40,80], roll_windows=[3,5,10,20]):
    D = df.copy()
    # target lags
    for l in lags:
        D[f'lag_{l}'] = D['target'].shift(l)
    # rolling stats of target (shifted to avoid leakage)
    for w in roll_windows:
        D[f'roll_mean_{w}'] = D['target'].shift(1).rolling(window=w, min_periods=1).mean()
        D[f'roll_std_{w}']  = D['target'].shift(1).rolling(window=w, min_periods=1).std().fillna(0)
    # lagged features of X itself
    for f in feat_cols:
        for l in [1,2,3,5]:
            D[f"{f}_lag{l}"] = D[f].shift(l)
    D = D.dropna().reset_index(drop=True)
    return D

fe = make_features(df)
print("Features built. shape:", fe.shape)
# Save features for debugging
fe.iloc[:5].to_csv(OUT_DIR/"fe_preview.csv", index=False)

# ----------------- Train/val split (last 15% as val) -----------------
val_frac = 0.15
split_idx = int(len(fe)*(1-val_frac))
train_df = fe.iloc[:split_idx].reset_index(drop=True)
val_df   = fe.iloc[split_idx:].reset_index(drop=True)
print("Train rows:", len(train_df), "Val rows:", len(val_df))

# X/y for LightGBM (tabular)
tab_features = [c for c in train_df.columns if c not in ['target','target_smooth']]
X_tr_tab, y_tr_tab = train_df[tab_features].values, train_df['target'].values
X_val_tab, y_val_tab = val_df[tab_features].values, val_df['target'].values

# ------------- LightGBM (strong baseline) -------------
print("Training LightGBM (tuned baseline)...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED
)
# fit with early stopping if supported
try:
    lgb_model.fit(X_tr_tab, y_tr_tab, eval_set=[(X_val_tab, y_val_tab)], early_stopping_rounds=100, verbose=200)
except TypeError:
    lgb_model.fit(X_tr_tab, y_tr_tab)

pred_lgb = lgb_model.predict(X_val_tab)
print("LightGBM val NSE:", nse(y_val_tab, pred_lgb), "RMSE:", rmse(y_val_tab, pred_lgb))
np.save(OUT_DIR/"pred_lgb.npy", pred_lgb)
# save model
import joblib
joblib.dump(lgb_model, OUT_DIR/"lgb_model.joblib")

# ------------- Prepare sequences for NN models -------------
# We'll build sequences of WINDOW length using past features + past target
# include columns: feat_cols + 'target' (past values) so model sees history
data_cols = feat_cols + ['target']
arr = fe[data_cols].values  # aligned after make_features()
# build sequences and next-step targets
seqs = []
targets = []
for i in range(WINDOW, len(arr)):
    seqs.append(arr[i-WINDOW:i])
    targets.append(fe['target'].iloc[i])  # predicting original target at time i
seqs = np.array(seqs)   # shape (N_windows, WINDOW, n_features)
targets = np.array(targets)
print("Sequences shape:", seqs.shape, "Targets shape:", targets.shape)

# split for training
split_seq = int(seqs.shape[0] * (1 - val_frac))
Xtr_seq, Xval_seq = seqs[:split_seq], seqs[split_seq:]
ytr_seq, yval_seq = targets[:split_seq], targets[split_seq:]
print("Seq train:", Xtr_seq.shape, "Seq val:", Xval_seq.shape)

# scale per-feature (fit on train)
# scale per-feature (fit on train)
n_feats = Xtr_seq.shape[2]
scalers = []
for f in range(n_feats):
    # Fit scaler on 1 column (flattened)
    s = StandardScaler().fit(Xtr_seq[:,:,f].reshape(-1,1))
    scalers.append(s)

    # Transform train
    Xtr_seq[:,:,f] = s.transform(
        Xtr_seq[:,:,f].reshape(-1,1)
    ).reshape(Xtr_seq[:,:,f].shape)

    # Transform val
    Xval_seq[:,:,f] = s.transform(
        Xval_seq[:,:,f].reshape(-1,1)
    ).reshape(Xval_seq[:,:,f].shape)

    Xtr_seq[:,:,f] = s.transform(
        Xtr_seq[:,:,f].reshape(-1,1)
    ).reshape(Xtr_seq[:,:,f].shape)

    # Transform val
    Xval_seq[:,:,f] = s.transform(
        Xval_seq[:,:,f].reshape(-1,1)
    ).reshape(Xval_seq[:,:,f].shape)

sy = StandardScaler().fit(ytr_seq.reshape(-1,1))
ytr_s = sy.transform(ytr_seq.reshape(-1,1)).ravel()
yval_s = sy.transform(yval_seq.reshape(-1,1)).ravel()

# Save scalers
joblib.dump(scalers, OUT_DIR/"scalers.joblib")
joblib.dump(sy, OUT_DIR/"scaler_y.joblib")

# ------------- PyTorch datasets -------------
class SeqDataset(Dataset):
    def __init__(self, X, y): 
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(SeqDataset(Xtr_seq, ytr_s), batch_size=BATCH, shuffle=True, drop_last=True)
val_loader   = DataLoader(SeqDataset(Xval_seq, yval_s), batch_size=BATCH, shuffle=False)

# ------------- Transformer Regressor -------------
# Simple Transformer encoder + head for regression
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)  # 1 x max_len x d_model
    def forward(self, x):
        # x: batch x seq_len x d_model
        return x + self.pe[:, :x.size(1), :].to(x.device)

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, dim_feedforward=256, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model, max_len=2000)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,1))
    def forward(self, x):
        x = self.input_proj(x)    # batch x seq x d_model
        x = self.pos(x)
        x = self.transformer(x)   # batch x seq x d_model
        x = x[:, -1, :]           # take last token's representation
        return self.head(x).squeeze()

# instantiate
model = TransformerRegressor(input_dim=n_feats, d_model=128, nhead=8, num_layers=3).to(DEVICE)
opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(opt, factor=0.5, patience=4, verbose=True)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type=='cuda'))

# training loop with early stopping
best_val = -1e9
patience = 8
wait = 0
for epoch in range(1, EPOCHS+1):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=(DEVICE.type=='cuda')):
            out = model(xb)
            loss = ((out - yb)**2).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        train_losses.append(loss.item())
    # validation
    model.eval()
    preds = []
    trues = []
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            vloss = ((out - yb)**2).mean()
            val_losses.append(vloss.item())
            preds.append(out.cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds = np.concatenate(preds).ravel()
    trues = np.concatenate(trues).ravel()
    preds_orig = sy.inverse_transform(preds.reshape(-1,1)).ravel()
    trues_orig = sy.inverse_transform(trues.reshape(-1,1)).ravel()
    val_nse = nse(trues_orig, preds_orig)
    val_rmse = rmse(trues_orig, preds_orig)
    mean_train_loss = float(np.mean(train_losses))
    mean_val_loss = float(np.mean(val_losses))
    print(f"Epoch {epoch} train_loss {mean_train_loss:.6f} val_loss {mean_val_loss:.6f} val_nse {val_nse:.4f} val_rmse {val_rmse:.4f}")
    scheduler.step(mean_val_loss)
    # early stopping
    if val_nse > best_val + 1e-4:
        best_val = val_nse
        wait = 0
        torch.save(model.state_dict(), OUT_DIR/"transformer_best.pth")
    else:
        wait += 1
        if wait > patience:
            print("Early stopping transformer at epoch", epoch)
            break

# load best
model.load_state_dict(torch.load(OUT_DIR/"transformer_best.pth", map_location=DEVICE))
# predict on val set fully
model.eval()
all_preds = []
with torch.no_grad():
    for xb, _ in val_loader:
        xb = xb.to(DEVICE)
        out = model(xb).cpu().numpy()
        all_preds.append(out)
all_preds = np.concatenate(all_preds).ravel()
pred_tr = sy.inverse_transform(all_preds.reshape(-1,1)).ravel()
print("Transformer final val NSE:", nse(yval_seq, pred_tr), "RMSE:", rmse(yval_seq, pred_tr))
np.save(OUT_DIR/"pred_transformer.npy", pred_tr)

# ------------- LSTM baseline (lightweight) -------------
class LSTMReg(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(nn.Linear(hidden,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self,x):
        out,_ = self.lstm(x)
        out = out[:,-1,:]
        return self.fc(out).squeeze()

lstm_model = LSTMReg(n_feats, hidden=128).to(DEVICE)
opt2 = AdamW(lstm_model.parameters(), lr=1e-3)
scheduler2 = ReduceLROnPlateau(opt2, factor=0.5, patience=4, verbose=True)
scaler2 = torch.cuda.amp.GradScaler(enabled=(DEVICE.type=='cuda'))

best_val_lstm = -1e9; wait=0
for epoch in range(1, 31):
    lstm_model.train()
    for xb,yb in train_loader:
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        opt2.zero_grad()
        with torch.cuda.amp.autocast(enabled=(DEVICE.type=='cuda')):
            out = lstm_model(xb)
            loss = ((out - yb)**2).mean()
        scaler2.scale(loss).backward()
        scaler2.unscale_(opt2)
        torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
        scaler2.step(opt2); scaler2.update()
    # validation
    lstm_model.eval()
    preds=[]; trues=[]
    with torch.no_grad():
        for xb,yb in val_loader:
            xb=xb.to(DEVICE)
            p = lstm_model(xb).cpu().numpy()
            preds.append(p); trues.append(yb.numpy())
    preds = np.concatenate(preds).ravel()
    preds_orig = sy.inverse_transform(preds.reshape(-1,1)).ravel()
    truths_orig = sy.inverse_transform(np.concatenate(trues).reshape(-1,1)).ravel()
    val_nse_lstm = nse(truths_orig, preds_orig)
    print(f"LSTM epoch {epoch} val_nse {val_nse_lstm:.4f}")
    if val_nse_lstm > best_val_lstm + 1e-4:
        best_val_lstm = val_nse_lstm; wait=0
        torch.save(lstm_model.state_dict(), OUT_DIR/"lstm_best.pth")
    else:
        wait+=1
        if wait>6:
            break

# load best and save preds
lstm_model.load_state_dict(torch.load(OUT_DIR/"lstm_best.pth", map_location=DEVICE))
lstm_model.eval()
all_preds=[]
with torch.no_grad():
    for xb,_ in val_loader:
        xb=xb.to(DEVICE)
        all_preds.append(lstm_model(xb).cpu().numpy())
all_preds = np.concatenate(all_preds).ravel()
pred_lstm = sy.inverse_transform(all_preds.reshape(-1,1)).ravel()
np.save(OUT_DIR/"pred_lstm.npy", pred_lstm)
print("LSTM val NSE:", nse(yval_seq, pred_lstm), "RMSE:", rmse(yval_seq, pred_lstm))

# ------------- Stacking (meta-learner on validation) -------------
print("Running stacking (RidgeCV) on validation preds...")
preds_stack = np.vstack([
    np.load(OUT_DIR/"pred_lgb.npy") if (OUT_DIR/"pred_lgb.npy").exists() else pred_lgb,
    np.load(OUT_DIR/"pred_transformer.npy"),
    np.load(OUT_DIR/"pred_lstm.npy"),
    np.load(OUT_DIR/"pred_lgb.npy")  # duplicate as placeholder if needed
]).T

# align shapes: pred_lgb was created for fe-val; ensure lengths match
y_stack_truth = y_val_tab[-preds_stack.shape[0]:] if preds_stack.shape[0] < len(y_val_tab) else y_val_tab[:preds_stack.shape[0]]
# meta learner
alphas = [0.1,1.0,10.0,100.0]
meta = RidgeCV(alphas=alphas, cv=3).fit(preds_stack, y_stack_truth)
stack_pred = meta.predict(preds_stack)
print("Stack NSE:", nse(y_stack_truth, stack_pred), "RMSE:", rmse(y_stack_truth, stack_pred))
np.save(OUT_DIR/"stack_pred.npy", stack_pred)

# Save meta and config
joblib.dump(meta, OUT_DIR/"stack_meta.joblib")
with open(OUT_DIR/"config.json","w") as f:
    json.dump({"WINDOW":WINDOW,"BATCH":BATCH,"EPOCHS":EPOCHS,"device":str(DEVICE)}, f)

print("Pipeline completed. Outputs saved to", str(OUT_DIR))
