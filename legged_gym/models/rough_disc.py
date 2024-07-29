import torch
import torch.nn as nn


class MLPDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPDiscriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim, input_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Pass the input through the LSTM layer
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_dim)

        # Global average pooling over the batch and sequence dimensions
        avg_lstm_out = torch.mean(lstm_out, dim=(0, 1))  # avg_lstm_out: (hidden_dim)

        # Pass the pooled output through the fully connected layers
        out = self.fc(avg_lstm_out)  # out: (output_dim)
        return out
