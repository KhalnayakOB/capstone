# snn_model.py

import torch
import torch.nn as nn


class SpikeFn(torch.autograd.Function):
    """
    Surrogate gradient spike function: hard threshold in forward,
    triangular derivative in backward.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # piecewise linear around 0
        sigma = 1.0
        grad = grad_output.clone()
        mask = (x.abs() < sigma).float()
        return grad * mask * (0.5 / sigma)


spike_fn = SpikeFn.apply


class LIFLayer(nn.Module):
    def __init__(self, in_features, out_features,
                 tau=20.0, v_th=1.0):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.tau = tau
        self.v_th = v_th

    def forward(self, x, v):
        """
        x: (batch, in_features)
        v: (batch, out_features) membrane voltage
        Returns: spikes, new_v
        """
        I = self.fc(x)
        dv = (-v + I) / self.tau
        v = v + dv
        s = spike_fn(v - self.v_th)
        # reset where spike occurred
        v = v * (1.0 - s)
        return s, v


class SpikingPolicyNet(nn.Module):
    """
    Multi-layer SNN:
      input_dim -> hidden_dim -> hidden_dim -> output_dim

    We unroll for T time steps and return the average spike rate
    of the last layer as continuous output.
    """

    def __init__(self, input_dim=6, hidden_dim=64, output_dim=3, T=15):
        super().__init__()
        self.T = T

        self.lif1 = LIFLayer(input_dim, hidden_dim)
        self.lif2 = LIFLayer(hidden_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: (batch, input_dim)
        Returns: (batch, output_dim) continuous velocities
        """
        batch_size = x.size(0)
        v1 = torch.zeros(batch_size, self.lif1.fc.out_features, device=x.device)
        v2 = torch.zeros(batch_size, self.lif2.fc.out_features, device=x.device)

        spike_sum = torch.zeros_like(v2)

        for _ in range(self.T):
            s1, v1 = self.lif1(x, v1)
            s2, v2 = self.lif2(s1, v2)
            spike_sum += s2

        # average spike rate
        r = spike_sum / self.T
        out = self.readout(r)
        return out
