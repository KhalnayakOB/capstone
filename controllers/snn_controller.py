import numpy as np

class LIFNeuronLayer:
    def __init__(self, input_size, output_size, tau=0.02, dt=0.01):
        self.input_size = input_size
        self.output_size = output_size

        # Random small weights
        self.W = np.random.randn(output_size, input_size) * 0.1  

        # Membrane potentials
        self.v = np.zeros(output_size)
        self.tau = tau
        self.dt = dt

    def forward(self, x):
        """
        x: input vector (input_size)
        returns: spikes (0 or 1) for output_size
        """
        # LIF membrane update
        dv = (-self.v + np.dot(self.W, x)) * (self.dt / self.tau)
        self.v += dv

        # Threshold and spike
        spikes = (self.v > 1.0).astype(float)

        # Reset membrane on spike
        self.v[spikes == 1.0] = 0.0

        return spikes


class SNNController:
    def __init__(self, input_size=6, hidden_size=20, output_size=3, lr=0.01):
        self.layer1 = LIFNeuronLayer(input_size, hidden_size)
        self.layer2 = LIFNeuronLayer(hidden_size, output_size)
        self.lr = lr

    def forward(self, state_vec):
        # Convert state vector to numpy array
        x = np.array(state_vec, dtype=float)

        h = self.layer1.forward(x)
        o = self.layer2.forward(h)

        # Decode spikes into continuous control signals
        # Simple linear decoder:
        return o * 5.0   # scale the spikes into force values

    def train_step(self, state_vec, target):
        """
        Simple learning rule: nudges weights toward target control.
        target: np.array shape (3,)
        """
        # Forward pass
        x = np.array(state_vec, dtype=float)
        h = self.layer1.forward(x)
        o = self.layer2.forward(h)

        # Compute error
        error = target - o

        # Update second layer weights
        self.layer2.W += self.lr * np.outer(error, h)

        # Update first layer weights
        self.layer1.W += self.lr * np.outer(
            np.dot(self.layer2.W.T, error), x
        )

        return o
