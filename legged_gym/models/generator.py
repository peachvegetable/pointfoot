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
    def __init__(self, input_dim, hidden_dim, output_dim, output_range, num_layers=6, num_heads=8):
        super(TransformerGenerator, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, hidden_dim))
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, activation="relu")
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.output_range = output_range

    def forward(self, x):
        # Additional positional encoding
        x = self.embedding(x) + self.positional_encoding
        x = F.relu(x)
        x = x.unsqueeze(0)
        x = self.transformer_encoder(x)
        x = x.squeeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_out(x))
        for i, (min_val, max_val) in enumerate(self.output_range):
            x[0][i] = torch.sigmoid(x[0][i]) * (max_val - min_val) + min_val
        return x
