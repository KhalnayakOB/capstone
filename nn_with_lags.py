# nn_with_lags.py
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def nse(y,yhat):
    num = ((y-yhat)**2).sum()
    den  = ((y-y.mean())**2).sum()
    return 1 - num/den

npz = np.load("C:/Users/bhava/OneDrive/Desktop/capstone/expert_data_rrt_3d_v2_1000_noisy.npz")
X = npz['X']; Y = npz['Y'][:,0]
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
df['target'] = Y

# same feature builder as LightGBM (smaller lags ok)
def add_lags(df, lags=[1,2,3,5,10,20]):
    d = df.copy()
    for l in lags:
        d[f'lag_{l}'] = d['target'].shift(l)
    d['roll_mean_3'] = d['target'].shift(1).rolling(3,min_periods=1).mean()
    d['roll_std_3']  = d['target'].shift(1).rolling(3,min_periods=1).std().fillna(0)
    return d.dropna().reset_index(drop=True)

fe = add_lags(df)
split = int(len(fe)*0.85)
train = fe.iloc[:split]; val = fe.iloc[split:]
features = [c for c in train.columns if c!='target']

scalerX = StandardScaler().fit(train[features])
scalery = StandardScaler().fit(train[['target']])

Xtr = torch.tensor(scalerX.transform(train[features]), dtype=torch.float32)
ytr = torch.tensor(scalery.transform(train[['target']]), dtype=torch.float32).squeeze()
Xv  = torch.tensor(scalerX.transform(val[features]), dtype=torch.float32)
yv  = torch.tensor(scalery.transform(val[['target']]), dtype=torch.float32).squeeze()

class TDataset(Dataset):
    def __init__(self,X,y): self.X=X; self.y=y
    def __len__(self): return len(self.y)
    def __getitem__(self,idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(TDataset(Xtr,ytr), batch_size=128, shuffle=True)
val_loader = DataLoader(TDataset(Xv,yv), batch_size=256, shuffle=False)

# simple MLP
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,1)
        )
    def forward(self,x): return self.net(x).squeeze()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(Xtr.shape[1]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.MSELoss()

# training loop
best_val = 1e9
for epoch in range(1,51):
    model.train()
    for xb,yb in train_loader:
        xb,yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward(); opt.step()
    # val
    model.eval()
    preds=[]; truths=[]
    with torch.no_grad():
        for xb,yb in val_loader:
            xb=xb.to(device)
            p = model(xb).cpu().numpy()
            preds.append(p); truths.append(yb.numpy())
    preds = np.concatenate(preds).ravel()
    truths = np.concatenate(truths).ravel()
    preds_orig = scalery.inverse_transform(preds.reshape(-1,1)).ravel()
    truths_orig = scalery.inverse_transform(truths.reshape(-1,1)).ravel()
    val_mse = mean_squared_error(truths_orig, preds_orig)
    val_nse = nse(truths_orig, preds_orig)
    print(f"Epoch {epoch} val_mse {val_mse:.4f} val_nse {val_nse:.4f}")

print("Done. Final NN NSE:", val_nse)
