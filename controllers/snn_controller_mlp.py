import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class MultiLayerSNN(nn.Module):
    """
    Deep multi-layer SNN for UAV control:
      - Input: 6  (pos_x, pos_y, pos_z, goal_x, goal_y, goal_z)
      - Hidden: 256 -> 256 -> 128 spiking LIF layers
      - Output: 3  (vx, vy, vz) via rate decoding
    """
    def __init__(self, num_steps: int = 10):
        super().__init__()

        self.num_steps = num_steps
        self.surrogate_grad = surrogate.fast_sigmoid()

        self.input_dim = 6
        self.h1 = 256
        self.h2 = 256
        self.h3 = 128
        self.output_dim = 3

        # Fully-connected layers
        self.fc1 = nn.Linear(self.input_dim, self.h1)
        self.fc2 = nn.Linear(self.h1, self.h2)
        self.fc3 = nn.Linear(self.h2, self.h3)
        self.fc_out = nn.Linear(self.h3, self.output_dim)

        # LIF spiking layers
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=self.surrogate_grad)
        self.lif2 = snn.Leaky(beta=0.85, spike_grad=self.surrogate_grad)
        self.lif3 = snn.Leaky(beta=0.8, spike_grad=self.surrogate_grad)

    def forward(self, x: torch.Tensor, num_steps: int = None) -> torch.Tensor:
        """
        x: (batch_size, 6)
        num_steps: simulation timesteps for temporal SNN processing
        returns: (batch_size, 3) continuous velocities
        """
        if num_steps is None:
            num_steps = self.num_steps

        batch_size = x.size(0)
        device = x.device

        # Initialize membrane potentials manually as zeros
        mem1 = torch.zeros(batch_size, self.h1, device=device)
        mem2 = torch.zeros(batch_size, self.h2, device=device)
        mem3 = torch.zeros(batch_size, self.h3, device=device)

        out_sum = 0.0

        for t in range(num_steps):
            # Encode input as constant current at each timestep
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            out = self.fc_out(spk3)
            out_sum = out_sum + out

        # Average over timesteps → rate-coded continuous output
        return out_sum / num_steps


def load_trained_snn(path: str = "snn_rrt_3d_model.pt") -> MultiLayerSNN:
    model = MultiLayerSNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
