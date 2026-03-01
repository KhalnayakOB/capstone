import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from snn_controller_mlp import MultiLayerSNN


def load_expert_dataset(path="expert_data_rrt.npz"):
    data = np.load(path)
    X = torch.tensor(data["X"], dtype=torch.float32)
    Y = torch.tensor(data["Y"], dtype=torch.float32)
    return X, Y


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training device:", device)

    X, Y = load_expert_dataset()
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = MultiLayerSNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    EPOCHS = 25

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            out = model(xb, num_steps=10)

            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch}/{EPOCHS} - loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "snn_rrt_model.pt")
    print("Saved trained multi-layer SNN to snn_rrt_model.pt")


if __name__ == "__main__":
    train()
