import torch
import torch.nn as nn


class MLPGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, output_range):
        super(MLPGenerator, self).__init__()
        self.output_dim = output_dim
        self.output_range = output_range
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        for i, (min_val, max_val) in enumerate(self.output_range):
            x[i] = x[i] * (max_val - min_val) + min_val
        return x
