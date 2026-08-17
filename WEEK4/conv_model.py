import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ConvDQN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv_layers(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv_layers(x)
        conv_out = conv_out.view(x.size(0), -1)
        return self.fc_layers(conv_out) 