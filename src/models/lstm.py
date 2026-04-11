import torch.nn as nn

class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, H=20):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, H)

    def forward(self, x):
        # x: (B, L)
        x = x.unsqueeze(-1)  # → (B, L, 1)

        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]  # (B, hidden)

        y_hat = self.fc(last_hidden)  # (B, H)
        return y_hat