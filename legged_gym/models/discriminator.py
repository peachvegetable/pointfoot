import torch
import torch.nn as nn


class TransformerDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, num_heads=4):
        super(TransformerDiscriminator, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = x.permute(1, 0, 2)  # reorder to (sequence_length, batch_size, hidden_dim)
        out = self.transformer_encoder(x)  # output is (sequence_length, batch_size, hidden_dim)
        out = out.permute(1, 0, 2)  # reorder to (batch_size, sequence_length, hidden_dim)
        out = self.fc2(out[:, -1, :])  # Take the output of the last time step
        out = self.sigmoid(out)
        return out
