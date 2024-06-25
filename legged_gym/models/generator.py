import torch
import torch.nn as nn
import torch.nn.functional as F


# class Generator(nn.Module):
#     def __init__(self, noise_dim, hidden_dim, output_range):
#         super(Generator, self).__init__()
#         self.fc1 = nn.Linear(noise_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, 1)
#         self.output_range = output_range
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.tanh(self.fc3(x))
#         x = (x + 1) / 2
#         x = x * (self.output_range[1] - self.output_range[0]) + self.output_range[0]
#         return x

class TransformerGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, num_heads=4):
        super(TransformerGenerator, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, hidden_dim))
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Additional positional encoding
        x = self.embedding(x) + self.positional_encoding
        x = x.unsqueeze(0)  # Adding sequence dimension
        x = self.transformer_encoder(x)
        x = x.squeeze(0)  # Removing sequence dimension
        x = torch.tanh(self.fc_out(x))
        return x
