import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

data = pd.read_csv("dataset_rrt.csv")

X = data[["px","py","gx","gy","dist","obs_dist","obs_dx","obs_dy"]].values
y = data[["dx","dy"]].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

class SNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 2),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

model = SNN().to(device)

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

best_loss = float("inf")
patience = 50
counter = 0

for epoch in range(800):

    model.train()
    pred = model(X_train)
    loss = criterion(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    if loss.item() < best_loss:
        best_loss = loss.item()
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        counter += 1

    if counter > patience:
        print("Early stopping triggered")
        break

model.load_state_dict(torch.load("best_model.pth"))
model.eval()

with torch.no_grad():
    test_loss = criterion(model(X_test), y_test)

print(f"\nTest Loss: {test_loss.item():.6f}")
print("✅ Model saved")