import numpy as np
from src.models.blocks.conv_block import ConvBlock
from src.models.blocks.fully_connected import FullyConnected
from src.settings import Settings
import torch
from torch import nn
from typing import List


class HEAD(nn.Module):
    def __init__(self, input_sizes: List, settings: Settings) -> None:
        super(HEAD, self).__init__()
        self._settings = settings

        # DO NOT CHANGE HERE, CHANGE IN _define_refine
        self._repeats = None
        self._channel_div = None
        self._stride = None
        self._padding = None
        self._linear_out = None
        self._define_refine()

        self._rnn = nn.GRU

        self._refine = nn.ModuleList([self._make_refine(x[0]) for x in input_sizes])
        self._rnn = self._rnn(
            input_size = settings._lstm_input_size,
            hidden_size = settings.lstm_hidden_size,
            num_layers = settings.lstm_num_layers,
            bias = settings.lstm_bias,
            dropout = settings.lstm_dropout_prob,
            bidirectional = settings.lstm_bidirectional,
            batch_first=True
            )
        self._classifier = nn.Linear(
            in_features=settings.lstm_hidden_size if not settings.lstm_bidirectional else settings.lstm_hidden_size * 2,
            out_features=len(settings.dataset_labels),
            bias=True
            )
        
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        bs, seq = x[0].shape[0], x[0].shape[1]
        x = [refine_module(x[i].view(-1, x[i].shape[2], x[i].shape[3], x[i].shape[4])) for i, refine_module in enumerate(self._refine)]
        x = [xi.view(bs, seq, -1) for xi in x]
        x_flat = torch.Tensor([]).to(self._settings.train_dev)
        for xi in x:
            x_flat = torch.cat((x_flat, xi), dim=2)
        x, _ = self._rnn(x_flat)
        x = self._classifier(x[:,-1,:])
        return x
    
    def _define_refine(self) -> np.ndarray:
        """
        Refinement layers:

            repeat x ConvBlock (3x3 kernel)
            1x Conv2d (1x1 kernel)
            fc (1x)

        example: repeat = [2, 1]
        -> (ConvBlock, ConvBlock), Conv2d, (ConvBlock), Conv2d, fc
        -> channel_div, stride, padding MUST BE shape (2, 1)
        """
        self._repeats = [2]
        self._channel_div = [[0.5, 0.5]]
        self._stride = [[2, 2]]
        self._padding = [[0, 0]]
        self._fc_in = 1024 # TODO: don't hardcode

    def _make_refine(self, in_channels: int) -> nn.Sequential:
        assert len(self._repeats) > 0, "At least 1 repetition needed"
        assert len(self._stride) == len(self._padding) == len(self._channel_div) == len(self._repeats)

        sequential = nn.Sequential()
        for i, r in enumerate(self._repeats):
            for j in range(r):
                out_channels = np.int64(in_channels // self._channel_div[i][j]) # fp doesn't work with pytorch
                sequential.append(ConvBlock(in_channels, out_channels, (3,3), self._stride[i][j], self._padding[i][j], activation_function=nn.SiLU(True)))
                in_channels = out_channels
            # sequential.append(nn.Conv2d(in_channels, in_channels, (1,1), 1, 0, bias=False))
        sequential.append(FullyConnected(self._fc_in, self._settings.lstm_input_size, True))
        return sequential