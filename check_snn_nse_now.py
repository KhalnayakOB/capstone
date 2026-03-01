import numpy as np
import torch
from controllers.snn_controller_deep import MultiLayerDeepSNN

def nse(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

# ---------------- LOAD DATA ----------------
# Use the SAME dataset used during SNN training
data = np.load("data/processed/expert_data_rrt_3d_v2.npz")

X = data["X"]   # shape (N, 6)
Y = data["Y"]   # shape (N, 3)

# ---------------- LOAD MODEL ----------------
model = MultiLayerDeepSNN()
model.load_state_dict(torch.load("models/snn_deep_v2.pt", map_location="cpu"))
model.eval()

# ---------------- PREDICT ----------------
with torch.no_grad():
    preds = model(torch.tensor(X, dtype=torch.float32)).numpy()

# ---------------- NSE ----------------
nse_x = nse(Y[:, 0], preds[:, 0])
nse_y = nse(Y[:, 1], preds[:, 1])
nse_z = nse(Y[:, 2], preds[:, 2])
nse_avg = (nse_x + nse_y + nse_z) / 3

print(f"SNN NSE vx = {nse_x:.4f}")
print(f"SNN NSE vy = {nse_y:.4f}")
print(f"SNN NSE vz = {nse_z:.4f}")
print(f"SNN NSE (avg) = {nse_avg:.4f}")
