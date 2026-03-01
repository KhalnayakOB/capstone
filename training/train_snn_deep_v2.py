# training/train_snn_deep_v2.py
"""
Train deep 7-layer SNN on V2 noisy RRT* dataset.
- Expects dataset at: data/processed/expert_data_rrt_3d_v2_1000_noisy.npz
- Saves best model to: models/snn_deep_v2.pt
"""

import os
import sys
import glob
import importlib.util
import time
from typing import Tuple

# -----------------------
# Make project root importable
# -----------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# -----------------------
# Robust import for controller module (works even if files moved)
# -----------------------
try:
    from controllers.snn_controller_deep import MultiLayerDeepSNN
except Exception:
    matches = glob.glob(os.path.join(ROOT, "**", "snn_controller_deep.py"), recursive=True)
    if not matches:
        raise ImportError(
            "Could not find 'snn_controller_deep.py' in project. "
            "Run: Get-ChildItem -Recurse -Filter snn_controller_deep.py to locate it."
        )
    controller_path = matches[0]
    spec = importlib.util.spec_from_file_location("controllers.snn_controller_deep", controller_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "MultiLayerDeepSNN"):
        raise ImportError(f"Loaded {controller_path} but MultiLayerDeepSNN not defined inside.")
    MultiLayerDeepSNN = getattr(module, "MultiLayerDeepSNN")

# -----------------------
# Standard imports
# -----------------------
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

# -----------------------
# Utils
# -----------------------
def load_dataset(path: str = "data/processed/expert_data_rrt_3d_v2_1000_noisy.npz") -> Tuple[torch.Tensor, torch.Tensor]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Run the collector to create it.")
    data = np.load(path)
    X = torch.tensor(data["X"], dtype=torch.float32)
    Y = torch.tensor(data["Y"], dtype=torch.float32)
    return X, Y

def NSE_torch(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    # returns NSE per-dataset (single scalar)
    num = torch.sum((pred - true) ** 2)
    den = torch.sum((true - torch.mean(true, dim=0)) ** 2)
    # protect against degenerate denominator
    if den == 0:
        return torch.tensor(0.0)
    return 1.0 - (num / den)

# -----------------------
# Training loop
# -----------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training device:", device)

    # Load dataset
    X, Y = load_dataset()
    N = X.shape[0]
    print(f"Loaded dataset: samples={N}, X_dim={X.shape[1]}, Y_dim={Y.shape[1]}")

    # Shuffle + 70/15/15 split
    perm = torch.randperm(N)
    n_train = int(0.70 * N)
    n_val = int(0.15 * N)
    n_test = N - n_train - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    print(f"Split sizes -> Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Dataloaders
    batch_size = 128
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=512, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=512, shuffle=False)

    # Model, optimizer, loss
    model = MultiLayerDeepSNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Training hyperparams
    EPOCHS = 120
    best_val_nse = -1e9
    best_state = None
    best_epoch = -1
    patience = 12
    no_improve = 0

    # ensure models dir exists
    Path("models").mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        # ----- train -----
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            out = model(xb, num_steps=15)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        avg_train_loss = running_loss / float(len(train_loader.dataset))

        # ----- validation -----
        model.eval()
        with torch.no_grad():
            val_losses = []
            preds = []
            trues = []
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb, num_steps=15)
                val_losses.append(loss_fn(out, yb).item() * xb.size(0))
                preds.append(out.cpu())
                trues.append(yb.cpu())

            val_loss = float(sum(val_losses) / float(len(val_loader.dataset)))
            pred_cat = torch.cat(preds, dim=0)
            true_cat = torch.cat(trues, dim=0)
            val_nse = float(NSE_torch(pred_cat, true_cat).item())

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}/{EPOCHS} - train_loss: {avg_train_loss:.6f}  val_loss: {val_loss:.6f}  val_NSE: {val_nse:.4f}  time: {epoch_time:.1f}s")

        # Save best
        if val_nse > best_val_nse + 1e-6:
            best_val_nse = val_nse
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
            torch.save(best_state, "models/snn_deep_v2_best_temp.pt")
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= patience:
            print(f"No improvement for {patience} epochs — early stopping.")
            break

    total_time = time.time() - start_time
    print(f"Training finished in {total_time/60.0:.2f} minutes. Best val_NSE={best_val_nse:.4f} at epoch {best_epoch}")

    # Save final best model
    if best_state is not None:
        torch.save(best_state, "models/snn_deep_v2.pt")
        print("Saved best model to models/snn_deep_v2.pt")
    else:
        print("Warning: best_state is None — saving last model state.")
        torch.save(model.state_dict(), "models/snn_deep_v2.pt")

    # ----- Final test evaluation -----
    # load best state
    state = torch.load("models/snn_deep_v2.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    with torch.no_grad():
        test_losses = []
        preds = []
        trues = []
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb, num_steps=15)
            test_losses.append(loss_fn(out, yb).item() * xb.size(0))
            preds.append(out.cpu())
            trues.append(yb.cpu())

        test_loss = float(sum(test_losses) / float(len(test_loader.dataset)))
        pred_cat = torch.cat(preds, dim=0)
        true_cat = torch.cat(trues, dim=0)
        test_nse = float(NSE_torch(pred_cat, true_cat).item())

    print("\nFinal test performance:")
    print(f"Test MSE: {test_loss:.6f}")
    print(f"Test NSE: {test_nse:.6f}")


if __name__ == "__main__":
    train()
