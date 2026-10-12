import torch
import torch.nn as nn

"""Creates a multi-layer perceptron (MLP) model for regression tasks.
3 stages of linear layers with ReLU in between.
ReLU just turns negative values of y to 0 and keeps positive values as is. It is Nonlinear.
Linear gives weight and bias parameters.
Multilayer with ReLU helps model learn nonlinear relationships as it keeps filtering out negatives.
Otherwise, without ReLU this would be equivalent to a single layer linear model."""
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)