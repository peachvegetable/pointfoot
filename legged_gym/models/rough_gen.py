import torch
import torch.nn as nn


class MLPGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, output_range):
        super(MLPGenerator, self).__init__()
        self.output_size = output_dim
        self.output_range = output_range
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Tanh activation to generate data in range [-1, 1]
        )

    def forward(self, x):
        x = self.model(x)
        # for i, (min_val, max_val) in enumerate(self.output_range):
        #     x[i] = torch.sigmoid(x[i]) * (max_val - min_val) + min_val
        return x
