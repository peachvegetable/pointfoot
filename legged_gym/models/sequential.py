import torch
import torch.nn as nn


class LSTMDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, output_range):
        super(LSTMDiscriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim, input_dim, batch_first=True)
        self.output_range = output_range
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, output_dim)
        )
        self._initialize_weights()

    def forward(self, x):
        # Pass the input through the LSTM layer
        lstm_out, _ = self.lstm(x)

        avg_out = torch.mean(lstm_out, dim=0)

        # Pass the pooled output through the fully connected layers
        out = self.fc(avg_out)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', param=0.2))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=nn.init.calculate_gain('leaky_relu', param=0.2))
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
