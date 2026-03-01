import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

def nse(y,yhat):
    return 1 - ((y - yhat)**2).sum() / ((y - y.mean())**2).sum()

NPZ_PATH = r"C:\Users\bhava\OneDrive\Desktop\capstone\data\processed\expert_data_rrt_3d_v2_1000_noisy.npz"
npz = np.load(NPZ_PATH)
X = npz["X"]
Y = npz["Y"][:,0]

df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
df["target"] = Y

WINDOW = 60

sequences=[]
targets=[]
for i in range(WINDOW, len(df)):
    seq = df.iloc[i-WINDOW:i].values
    sequences.append(seq)
    targets.append(df["target"].iloc[i])

sequences = np.array(sequences)
targets = np.array(targets)

split = int(len(sequences)*0.85)
Xtr, Xv = sequences[:split], sequences[split:]
ytr, yv = targets[:split], targets[split:]

# scale per feature
scalers=[]
for f in range(Xtr.shape[2]):
    s = StandardScaler().fit(Xtr[:,:,f])
    Xtr[:,:,f] = s.transform(Xtr[:,:,f])
    Xv[:,:,f]  = s.transform(Xv[:,:,f])
    scalers.append(s)

sy = StandardScaler().fit(ytr.reshape(-1,1))
ytr_s = sy.transform(ytr.reshape(-1,1)).flatten()
yv_s  = sy.transform(yv.reshape(-1,1)).flatten()

class SeqDS(Dataset):
    def __init__(self,X,y): self.X=torch.tensor(X,dtype=torch.float32); self.y=torch.tensor(y,dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.X[i], self.y[i]

train_dl = DataLoader(SeqDS(Xtr,ytr_s), batch_size=64, shuffle=True)
val_dl   = DataLoader(SeqDS(Xv,yv_s), batch_size=128)

class LSTMReg(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self,x):
        out,_ = self.lstm(x)
        out = out[:,-1,:]
        return self.fc(out).flatten()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LSTMReg(Xtr.shape[2]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

best_nse = -999
for epoch in range(20):
    model.train()
    for xb,yb in train_dl:
        xb,yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward(); opt.step()

    # validation
    model.eval()
    preds=[]
    with torch.no_grad():
        for xb,_ in val_dl:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
    preds = np.concatenate(preds)
    preds_orig = sy.inverse_transform(preds.reshape(-1,1)).flatten()
    truth_orig = sy.inverse_transform(yv_s.reshape(-1,1)).flatten()

    score = nse(truth_orig, preds_orig)
    print(f"Epoch {epoch+1} NSE: {score:.4f}")
    best_nse = max(best_nse, score)

np.save("lstm_preds.npy", preds_orig)
print("Saved: lstm_preds.npy")
