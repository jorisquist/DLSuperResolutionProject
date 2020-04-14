import torch.nn as nn


class SuperResolutionNet(nn.Module):
    def __init__(self, r, activation=nn.Identity()):
        super().__init__()
        self.r = r
        self.activation = activation

        # self.conv1 = nn.Conv2d(3, 64, 5, padding=2)
        # self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        # self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        # self.conv4 = nn.Conv2d(32, self.r * self.r * 3, 3, padding=1)
        self.deconvolution = nn.PixelShuffle(self.r)

        self.layers = nn.ModuleList([
            nn.Conv2d(3, 64, 5, padding=2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.Conv2d(32, self.r * self.r * 3, 3, padding=1)
        ])

        self.params = list(self.layers.parameters())

        self.l = len(self.layers)  # The number of hidden layers

    def forward(self, x):
        for i in range(self.l - 1):
            x = self.activation(self.layers[i](x))

        x = self.layers[self.l - 1](x)  # Don't use the activation on the last convolutional layer
        x = self.deconvolution(x)

        return x
