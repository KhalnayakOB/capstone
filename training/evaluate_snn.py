import numpy as np
import torch
from snn_model import SpikingPolicyNet


def nse_score(y_true, y_pred):
    """
    Computes Nash-Sutcliffe Efficiency for multi-output regression.
    y_true, y_pred: numpy arrays of shape (N, 3)
    Returns: NSE per dimension + mean NSE
    """
    nse_values = []
    for i in range(y_true.shape[1]):
        t = y_true[:, i]
        p = y_pred[:, i]
        numerator = np.sum((t - p) ** 2)
        denominator = np.sum((t - np.mean(t)) ** 2)
        nse = 1 - numerator / (denominator + 1e-9)
        nse_values.append(nse)
    return np.array(nse_values), np.mean(nse_values)


def main():
    # Load dataset
    data = np.load("expert_data.npz")
    X = data["X"]
    Y = data["Y"]

    print(f"Dataset shapes: X={X.shape}, Y={Y.shape}")

    # Convert to torch
    X_t = torch.from_numpy(X).float()
    Y_t = torch.from_numpy(Y).float()

    # Load trained SNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpikingPolicyNet(input_dim=6, hidden_dim=64, output_dim=3, T=15)
    model.load_state_dict(torch.load("snn_model.pt", map_location=device))
    model.to(device)
    model.eval()

    # Run predictions
    preds = []
    with torch.no_grad():
        for i in range(len(X_t)):
            x = X_t[i].unsqueeze(0).to(device)
            y_hat = model(x).cpu().numpy()[0]
            preds.append(y_hat)

    preds = np.array(preds)
    y_true = Y

    # Compute NSE
    nse_each, nse_mean = nse_score(y_true, preds)

    print("\n===== NSE RESULTS =====")
    print(f"NSE vx : {nse_each[0]:.4f}")
    print(f"NSE vy : {nse_each[1]:.4f}")
    print(f"NSE vz : {nse_each[2]:.4f}")
    print(f"Mean NSE: {nse_mean:.4f}")
    print("=========================\n")


if __name__ == "__main__":
    main()
