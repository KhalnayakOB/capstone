import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from snn_controller_mlp import MultiLayerSNN


def load_dataset(path="expert_data_rrt_3d.npz"):
    data = np.load(path)
    X = torch.tensor(data["X"], dtype=torch.float32)
    Y = torch.tensor(data["Y"], dtype=torch.float32)
    return X, Y


def NSE(pred, true):
    return 1 - torch.sum((pred - true) ** 2) / torch.sum(
        (true - torch.mean(true, dim=0)) ** 2
    )


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training device:", device)

    X, Y = load_dataset()
    N = X.shape[0]
    print(f"Total samples: {N}")

    # 70-15-15 split
    indices = torch.randperm(N)
    n_train = int(0.70 * N)
    n_val = int(0.15 * N)
    n_test = N - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train), batch_size=64, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val), batch_size=256, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test, Y_test), batch_size=256, shuffle=False
    )

    model = MultiLayerSNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    EPOCHS = 100
    best_val_nse = -1e9
    best_state = None
    best_epoch = -1

    for epoch in range(1, EPOCHS + 1):
        # ---------- TRAIN ----------
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            out = model(xb, num_steps=10)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # ---------- VALIDATION ----------
        model.eval()
        with torch.no_grad():
            val_losses = []
            all_pred = []
            all_true = []
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb, num_steps=10)
                val_losses.append(loss_fn(out, yb).item())
                all_pred.append(out.cpu())
                all_true.append(yb.cpu())

            val_loss = float(np.mean(val_losses))
            pred_cat = torch.cat(all_pred, dim=0)
            true_cat = torch.cat(all_true, dim=0)
            val_nse = NSE(pred_cat, true_cat).item()

        print(f"Epoch {epoch}/{EPOCHS} - "
              f"train_loss: {epoch_loss:.4f}  "
              f"val_loss: {val_loss:.4f}  "
              f"val_NSE: {val_nse:.4f}")

        # Save best model by validation NSE
        if val_nse > best_val_nse:
            best_val_nse = val_nse
            best_state = model.state_dict()
            best_epoch = epoch

    # ---------- SAVE BEST ----------
    if best_state is not None:
        torch.save(best_state, "snn_rrt_3d_model.pt")
        print(f"\nSaved best model from epoch {best_epoch} with val_NSE={best_val_nse:.4f}")
    else:
        print("\nWARNING: No best model state captured (this should not happen).")

    # ---------- FINAL TEST EVALUATION ----------
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        all_pred = []
        all_true = []
        test_losses = []
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb, num_steps=10)
            test_losses.append(loss_fn(out, yb).item())
            all_pred.append(out.cpu())
            all_true.append(yb.cpu())

        test_loss = float(np.mean(test_losses))
        pred_cat = torch.cat(all_pred, dim=0)
        true_cat = torch.cat(all_true, dim=0)
        test_nse = NSE(pred_cat, true_cat).item()

    print("\nFinal test performance:")
    print(f"Test MSE: {test_loss:.6f}")
    print(f"Test NSE: {test_nse:.6f}")


if __name__ == "__main__":
    train()
