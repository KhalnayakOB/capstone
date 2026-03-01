import numpy as np
import torch
from controllers.snn_controller_mlp import MultiLayerSNN



def NSE(pred, true):
    return 1 - torch.sum((pred - true) ** 2) / torch.sum((true - torch.mean(true)) ** 2)


def evaluate(model_path="snn_rrt_3d_model.pt", dataset="data/processed/expert_data_rrt_3d_v2_1000_noisy.npz"):
    data = np.load(dataset)
    X = torch.tensor(data["X"], dtype=torch.float32)
    Y = torch.tensor(data["Y"], dtype=torch.float32)

    model = MultiLayerSNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        pred = model(X, num_steps=10)

    mse = torch.mean((pred - Y) ** 2)
    nse = NSE(pred, Y)

    print("Evaluation on 3D RRT* dataset:")
    print("MSE:", mse.item())
    print("NSE:", nse.item())


if __name__ == "__main__":
    evaluate()
