from src.settings import Settings
import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, settings: Settings):
        super(LSTM, self).__init__()
        self._settings = settings
        self._lstm = nn.LSTM(
            input_size = settings.lstm_input_size,
            hidden_size = settings.lstm_hidden_size,
            num_layers = settings.lstm_num_layers,
            bias = settings.lstm_bias,
            dropout = settings.lstm_dropout_prob,
            bidirectional = settings.lstm_bidirectional,
            batch_first=True
            )
        self._classifier = nn.Linear(
            in_features=settings.lstm_hidden_size,
            out_features=len(settings.dataset_labels),
            bias=True
            )
        
    def forward(self, x):
        for idx in range(x.shape[1]):
            out, _ = self._lstm(x[:,idx,:])

        out = self._classifier(out)
        return out
