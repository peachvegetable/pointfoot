import torch
import torch.nn as nn
import numpy as np


class MLPGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLPGenerator, self).__init__()

        layers = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims[2], hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims[0], output_dim),
            nn.LeakyReLU(0.2)
        ]

        self.model = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        x = self.model(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu', param=0.2))
                nn.init.constant_(m.bias, 0)
