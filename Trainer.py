import os
import sys
import signal
import numpy as np
import math
import time

import torch
from torch import nn, optim

from eval import evaluate


class ShutdownSignal:
    shut_down = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.shut_down = True


def mse_to_psnr(mse):
    return 10 * math.log10(1. / mse)

def train(net, use_gpu, train_loader, r,
          learning_rate=0.001,
          max_epochs_without_improvement=1000,
          max_epochs=100000):
    shutdown_signal = ShutdownSignal()
    if use_gpu:
        net = net.cuda()
        print('Running on gpu')

    loss_function = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate)

    computer_name = os.environ['COMPUTERNAME']

    lowest_loss = (0, float('inf'))
    highest_psnr = - float('inf')
    begin_time = time.time()
    minimum_psnr_to_save = 20
    for epoch in range(max_epochs):
        train_loss = []

        net.train()
        for input, target in train_loader:

            optimizer.zero_grad()

            output = net(input)

            loss = loss_function(output, target)

            loss.backward()

            optimizer.step()

            train_loss.append(loss.item())

        mean_train_loss = np.mean(train_loss)
        mean_psnr = mse_to_psnr(mean_train_loss)

        if mean_train_loss < lowest_loss[1]:
            print(f"Epoch: {epoch: >3} Training Loss: {mean_train_loss:.6f} Mean PSNR: {mean_psnr:.2f} in {time.time() - begin_time:.2f}s #")
            lowest_loss = (epoch, mean_train_loss)
            highest_psnr = mean_psnr

            if highest_psnr > minimum_psnr_to_save:
                torch.save(net, f'SuperResulutionNet_best_of_run-{computer_name}')

        elif epoch % 100 == 0:
            print(
                f"Epoch: {epoch: >3} in {time.time() - begin_time:.2f}s")

        if shutdown_signal.shut_down:
            # Doesn't work in windows when running from pycharm, but works when running from command line with ctrl+c
            print("Shutdown received...")
            break
        elif epoch > lowest_loss[0] + max_epochs_without_improvement:
            print(f"No improvement for the last {max_epochs_without_improvement} epochs, so stopping training...")
            break

    net.eval()
    if highest_psnr >= minimum_psnr_to_save:
        network_name = f'SuperResulutionNet_r-{r}_psnr-{int(round(highest_psnr * 100))}__mse-{int(round(lowest_loss[1] * 10000))}-{computer_name}'
        old_file = os.path.join(".", f'SuperResulutionNet_best_of_run-{computer_name}')
        new_file = os.path.join(".", network_name)
        print(f'Saving best epoch ({lowest_loss[0]}) with loss: {lowest_loss[1]} and psnr: {highest_psnr} as:')
        print(network_name)
        os.rename(old_file, new_file)

        print('Now evaluating the network.')
        evaluate(network_name)
    else:
        print("Not high enough psnr to save the network...")

    return lowest_loss[1]

