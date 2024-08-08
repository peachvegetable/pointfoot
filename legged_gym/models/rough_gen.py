import torch
import torch.nn as nn


class MLPGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, output_range):
        super(MLPGenerator, self).__init__()
        self.output_range = output_range
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.model(x)
        for i, (min_val, max_val) in enumerate(self.output_range):
            x[i] = torch.sigmoid(x[i]) * (max_val - min_val) + min_val
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu', param=0.2))
                nn.init.constant_(m.bias, 0)
