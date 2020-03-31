import math
import sys
from multiprocessing.spawn import freeze_support

import torch
from torch import optim, nn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
import numpy as np
import time
import os

from SuperResolutionDataset import SuperResolutionDataset
from SuperResolutionNet import SuperResolutionNet


def mse_to_psnr(mse):
    return 10 * math.log10(1. / mse)


def main():
    use_gpu = torch.cuda.is_available()
    bs = 64
    r = 3

    # Getting image data
    transform = transforms.Compose(
        [transforms.ToTensor()])  # ,

    training_set = SuperResolutionDataset('train_data/Set91', r, use_gpu=use_gpu)

    train_loader = torch.utils.data.DataLoader(training_set,
                                               batch_size=bs,
                                               shuffle=True,
                                               num_workers=0)

    net = SuperResolutionNet(r, activation=nn.ReLU())
    if use_gpu:
        net = net.cuda()
        print('Running on gpu')

    loss_function = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters())
    # scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    lowest_loss = (0, float('inf'))
    highest_psnr = - float('inf')
    max_epochs_without_improvement = 1000
    begin_time = time.time()
    for epoch in range(10000):
        train_loss = []
        psnr = []

        net.train()
        for input, target in train_loader:


            optimizer.zero_grad()

            output = net(input)

            loss = loss_function(output, target)

            loss.backward()

            optimizer.step()
            # scheduler.step(epoch)

            psnr.append(mse_to_psnr(loss.item()))
            train_loss.append(loss.item())

        mean_train_loss = np.mean(train_loss)


        if mean_train_loss < lowest_loss[1]:
            print(f"Epoch: {epoch: >3} Training Loss: {mean_train_loss:.6f} Mean PSNR: {np.mean(psnr):.3f} in {time.time() - begin_time:.2f}s #")
            lowest_loss = (epoch, mean_train_loss)
            highest_psnr = np.mean(psnr)
            if (highest_psnr > 25):
                computer_name = os.environ['COMPUTERNAME']
                torch.save(net, f'SuperResulutionNet_best_of_run-{computer_name}')
        elif epoch % 100 == 0:
            print(
                f"Epoch: {epoch: >3} in {time.time() - begin_time:.2f}s")

        if epoch > lowest_loss[0] + max_epochs_without_improvement:
            print(f"No improvement for the last {max_epochs_without_improvement} epochs, so stopping training...")
            net.eval()
            break

    computer_name = os.environ['COMPUTERNAME']
    old_file = os.path.join(".", "SuperResulutionNet_best_of_run")
    new_file = os.path.join(".",
                            f'SuperResulutionNet_r-{r}_psnr-{int(round(highest_psnr * 100))}__mse-{int(round(lowest_loss[1] * 10000))}-{computer_name}')
    print(
        f'Saving best epoch ({lowest_loss[0]}) with loss: {lowest_loss[1]} and psnr: {highest_psnr} as:\nSuperResulutionNet_r-{r}_psnr-{int(round(highest_psnr * 100))}__mse-{int(round(lowest_loss[1] * 10000))}')
    os.rename(old_file, new_file)



if __name__ == '__main__':
    freeze_support()
    main()
