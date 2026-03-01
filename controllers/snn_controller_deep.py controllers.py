import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class MultiLayerDeepSNN(nn.Module):
    """
    Deep 7-layer SNN for UAV control:

      Input:  6  (pos_x, pos_y, pos_z, goal_x, goal_y, goal_z)
      Hidden: 256 -> 256 -> 256 -> 256 -> 128 spiking layers
      Output: 3  (vx, vy, vz)

    Uses:
      - LIF neurons
      - Surrogate gradients (fast_sigmoid)
      - Temporal integration (num_steps timesteps)
    """

    def __init__(self, num_steps: int = 15):
        super().__init__()

        self.num_steps = num_steps
        self.surrogate_grad = surrogate.fast_sigmoid()

        self.input_dim = 6
        self.h1 = 256
        self.h2 = 256
        self.h3 = 256
        self.h4 = 256
        self.h5 = 128
        self.output_dim = 3

        # Dense layers
        self.fc1 = nn.Linear(self.input_dim, self.h1)
        self.fc2 = nn.Linear(self.h1, self.h2)
        self.fc3 = nn.Linear(self.h2, self.h3)
        self.fc4 = nn.Linear(self.h3, self.h4)
        self.fc5 = nn.Linear(self.h4, self.h5)
        self.fc_out = nn.Linear(self.h5, self.output_dim)

        # Spiking LIF layers
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=self.surrogate_grad)
        self.lif2 = snn.Leaky(beta=0.88, spike_grad=self.surrogate_grad)
        self.lif3 = snn.Leaky(beta=0.86, spike_grad=self.surrogate_grad)
        self.lif4 = snn.Leaky(beta=0.84, spike_grad=self.surrogate_grad)
        self.lif5 = snn.Leaky(beta=0.82, spike_grad=self.surrogate_grad)

    def forward(self, x: torch.Tensor, num_steps: int | None = None) -> torch.Tensor:
        """
        x: (batch_size, 6)
        returns: (batch_size, 3)
        """
        if num_steps is None:
            num_steps = self.num_steps

        batch_size = x.size(0)
        device = x.device

        # Initialize membrane potentials
        mem1 = torch.zeros(batch_size, self.h1, device=device)
        mem2 = torch.zeros(batch_size, self.h2, device=device)
        mem3 = torch.zeros(batch_size, self.h3, device=device)
        mem4 = torch.zeros(batch_size, self.h4, device=device)
        mem5 = torch.zeros(batch_size, self.h5, device=device)

        out_sum = 0.0

        for _ in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.fc4(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            cur5 = self.fc5(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)

            out = self.fc_out(spk5)
            out_sum = out_sum + out

        return out_sum / num_steps


def load_trained_deep_snn(path: str = "models/snn_deep_v2.pt") -> MultiLayerDeepSNN:
    model = MultiLayerDeepSNN()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
