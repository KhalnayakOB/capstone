# seq_model.py
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def nse(y,yhat):
    num = ((y-yhat)**2).sum(); den = ((y-y.mean())**2).sum(); return 1 - num/den

npz = np.load("C:/Users/bhava/OneDrive/Desktop/capstone/expert_data_rrt_3d_v2_1000_noisy.npz")
X = npz['X']; Y = npz['Y'][:,0]
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]); df['target']=Y

# build sequences: choose window
WINDOW = 80   # you can try 40, 80, 160
step = 1
cols = [c for c in df.columns]  # include features + target if you want
# We will use only past target and features as input
data = df.copy()
# normalize features and target (fit on training later)
# build sequences (X_seq shape [n_windows, WINDOW, n_features], y = next-step target)
seqs = []
ys = []
for i in range(WINDOW, len(data)):
    seq = data.iloc[i-WINDOW:i][['f0','f1','f2','f3','f4','f5','target']].values  # include past target as input
    seqs.append(seq)
    ys.append(data['target'].iloc[i])
seqs = np.array(seqs); ys = np.array(ys)

# train/val split
split = int(len(seqs)*0.85)
Xtr, Xv = seqs[:split], seqs[split:]
ytr, yv = ys[:split], ys[split:]

# scale per-feature (fit on Xtr flattened)
n_features = Xtr.shape[2]
scalers = []
for f in range(n_features):
    s = StandardScaler().fit(Xtr[:,:,f].reshape(-1,1))
    Xtr[:,:,f] = s.transform(Xtr[:,:,f])
    Xv[:,:,f]  = s.transform(Xv[:,:,f])
    scalers.append(s)
# scale y
sy = StandardScaler().fit(ytr.reshape(-1,1))
ytr_s = sy.transform(ytr.reshape(-1,1)).ravel()
yv_s = sy.transform(yv.reshape(-1,1)).ravel()

# dataloaders
import torch
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self,X,y): self.X=torch.tensor(X,dtype=torch.float32); self.y=torch.tensor(y,dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self,idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(SeqDataset(Xtr,ytr_s), batch_size=64, shuffle=True)
val_loader   = DataLoader(SeqDataset(Xv,yv_s), batch_size=128, shuffle=False)

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hid=128, nlayers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hid, nlayers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(nn.Linear(hid,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self,x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        return self.fc(out).squeeze()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(n_features, hid=128).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# training
for epoch in range(1,41):
    model.train()
    for xb,yb in train_loader:
        xb,yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward(); opt.step()
    # validation
    model.eval()
    preds=[]; trues=[]
    with torch.no_grad():
        for xb,yb in val_loader:
            xb=xb.to(device)
            p = model(xb).cpu().numpy()
            preds.append(p); trues.append(yb.numpy())
    preds = np.concatenate(preds).ravel(); trues = np.concatenate(trues).ravel()
    preds_orig = sy.inverse_transform(preds.reshape(-1,1)).ravel()
    trues_orig = sy.inverse_transform(trues.reshape(-1,1)).ravel()
    print(f"Epoch {epoch} val_nse {nse(trues_orig,preds_orig):.4f} rmse {mean_squared_error(trues_orig,preds_orig,squared=False):.4f}")
