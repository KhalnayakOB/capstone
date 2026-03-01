import numpy as np
import torch
from snn_controller_mlp import MultiLayerSNN


def NSE(pred, true):
    return 1 - torch.sum((pred - true) ** 2) / torch.sum((true - torch.mean(true)) ** 2)


def evaluate(model_path="snn_rrt_model.pt", dataset="expert_data_rrt.npz"):
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

    print("Evaluation results:")
    print("MSE:", mse.item())
    print("NSE:", nse.item())


if __name__ == "__main__":
    evaluate()
