import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def nse(y, yhat):
    return 1 - ((y - yhat)**2).sum() / ((y - y.mean())**2).sum()

NPZ_PATH = r"C:\Users\bhava\OneDrive\Desktop\capstone\data\processed\expert_data_rrt_3d_v2_1000_noisy.npz"
npz = np.load(NPZ_PATH)
X = npz["X"]
Y = npz["Y"][:,0]

df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
df["target"] = Y

def add_features(df):
    d = df.copy()
    for l in [1,2,3,5,10,20,40,80]:
        d[f"lag_{l}"] = d["target"].shift(l)
    return d.dropna().reset_index(drop=True)

fe = add_features(df)

split = int(len(fe)*0.85)
train = fe.iloc[:split]
val = fe.iloc[split:]

features = [c for c in fe.columns if c!="target"]

scX = StandardScaler().fit(train[features])
scY = StandardScaler().fit(train[["target"]])

Xtr = torch.tensor(scX.transform(train[features]), dtype=torch.float32)
ytr = torch.tensor(scY.transform(train[["target"]]), dtype=torch.float32).flatten()

Xv  = torch.tensor(scX.transform(val[features]), dtype=torch.float32)
yv  = torch.tensor(scY.transform(val[["target"]]), dtype=torch.float32).flatten()

class MyDataset(Dataset):
    def __init__(self, X,y): self.X=X; self.y=y
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.X[i], self.y[i]

train_dl = DataLoader(MyDataset(Xtr,ytr), batch_size=128, shuffle=True)
val_dl   = DataLoader(MyDataset(Xv,yv), batch_size=256)

class MLP(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n,128), nn.ReLU(),
            nn.Linear(128,64), nn.ReLU(),
            nn.Linear(64,1)
        )
    def forward(self,x): return self.net(x).flatten()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP(Xtr.shape[1]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

best_nse = -999
for epoch in range(30):
    model.train()
    for xb,yb in train_dl:
        xb,yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward(); opt.step()
    
    model.eval()
    preds=[]
    with torch.no_grad():
        for xb,_ in val_dl:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
    preds = np.concatenate(preds)
    preds_orig = scY.inverse_transform(preds.reshape(-1,1)).flatten()
    truth_orig = scY.inverse_transform(yv.reshape(-1,1)).flatten()
    
    score = nse(truth_orig, preds_orig)
    print(f"Epoch {epoch+1} NSE: {score:.4f}")
    best_nse = max(best_nse, score)

np.save("mlp_preds.npy", preds_orig)
print("Saved: mlp_preds.npy")
print("Final MLP NSE:", best_nse)
