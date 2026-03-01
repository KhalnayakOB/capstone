# train_snn.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from snn_model import SpikingPolicyNet


def main():
    data = np.load("expert_data.npz")
    X = data["X"]  # (N, 6)
    Y = data["Y"]  # (N, 3)

    print("Loaded expert_data.npz:")
    print("  X shape:", X.shape)
    print("  Y shape:", Y.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    X_t = torch.from_numpy(X).float()
    Y_t = torch.from_numpy(Y).float()

    dataset = TensorDataset(X_t, Y_t)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = SpikingPolicyNet(input_dim=6, hidden_dim=64, output_dim=3, T=15).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    n_epochs = 20  # you can increase later

    model.train()
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{n_epochs} - loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), "snn_model.pt")
    print("Saved trained SNN to snn_model.pt")


if __name__ == "__main__":
    main()
