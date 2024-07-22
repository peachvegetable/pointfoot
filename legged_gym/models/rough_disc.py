import torch
import torch.nn as nn


class MLPDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(MLPDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        x = x.view(batch_size * seq_len, input_dim)  # Flatten the sequence and batch dimensions
        out = self.model(x)
        out = out.view(batch_size, seq_len, -1)  # Reshape back to (batch_size, seq_len, 1)
        return out[:, -1, :]  # Use the output of the last time step
