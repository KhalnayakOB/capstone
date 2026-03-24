import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

data = pd.read_csv("dataset_rrt.csv")

X = data[["px","py","gx","gy","dist","obs_dist","obs_dx","obs_dy"]].values
y_true = data[["dx","dy"]].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

class SNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(512, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),

            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),

            torch.nn.Linear(128, 2),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

model = SNN().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

X = torch.tensor(X, dtype=torch.float32).to(device)

with torch.no_grad():
    y_pred = model(X).cpu().numpy()

def compute_angle_error(y_true, y_pred):
    dot = np.sum(y_true * y_pred, axis=1)
    norm_true = np.linalg.norm(y_true, axis=1)
    norm_pred = np.linalg.norm(y_pred, axis=1)

    cos_theta = dot / (norm_true * norm_pred + 1e-8)
    cos_theta = np.clip(cos_theta, -1, 1)

    return np.mean(np.arccos(cos_theta))

mse = np.mean((y_true - y_pred)**2)
mae = np.mean(np.abs(y_true - y_pred))
angle_deg = np.degrees(compute_angle_error(y_true, y_pred))

print("\n📊 MODEL PERFORMANCE")
print("----------------------")
print(f"MSE  : {mse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"Angle Error (deg): {angle_deg:.2f}")