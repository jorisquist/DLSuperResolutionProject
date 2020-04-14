import math
import sys
from multiprocessing.spawn import freeze_support

import torch
from ax import optimize
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
import numpy as np
import time
import os

import Trainer
from SuperResolutionDataset import SuperResolutionDataset
from SuperResolutionNet import SuperResolutionNet


import signal


from eval import evaluate


def main():

    best_parameters, best_values, experiment, model = optimize(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
        ],
        # Booth function
        evaluation_function=train_evaluate,
        minimize=True,
    )

    print(best_parameters, best_values, experiment, model)


def train_evaluate(parameters):
    use_gpu = torch.cuda.is_available()
    bs = 32
    r = 3

    training_set = SuperResolutionDataset('train_data/Set91', r, use_gpu=use_gpu)
    # training_set = SuperResolutionDataset('test_data/BSD500', r, use_gpu=use_gpu)

    train_loader = torch.utils.data.DataLoader(training_set,
                                               batch_size=bs,
                                               shuffle=True,
                                               num_workers=0)

    net = SuperResolutionNet(r, activation=nn.ReLU())
    return Trainer.train(net, use_gpu, train_loader, r, max_epochs=1000, max_epochs_without_improvement=100, learning_rate=parameters['lr'])


if __name__ == '__main__':
    freeze_support()
    main()
