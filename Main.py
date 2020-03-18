import math
import sys
from multiprocessing.spawn import freeze_support

import torch
from torch import optim, nn
from torchvision import transforms
import numpy as np

from SuperResolutionDataset import SuperResolutionDataset
from SuperResolutionNet import SuperResolutionNet


def mse_to_psnr(mse):
    return 10 * math.log10(1. / mse)


def main():
    use_gpu = torch.cuda.is_available()
    bs = 1
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
    optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-6, momentum=0.9, nesterov=True)

    lowest_loss = (0, float('inf'))
    for epoch in range(1000):
        train_loss = []
        psnr = []

        net.train()
        for input, target in train_loader:

            optimizer.zero_grad()

            output = net(input)

            loss = loss_function(output, target)
            psnr.append(mse_to_psnr(loss.item()))

            loss.backward()

            optimizer.step()

            train_loss.append(loss.item())

        mean_train_loss = np.mean(train_loss)
        print(f"Epoch: {epoch: >3} Training Loss: {mean_train_loss:.6f} Mean PSNR: {np.mean(psnr):.3f}")

        if mean_train_loss < lowest_loss[1]:
            lowest_loss = (epoch, mean_train_loss)

        if epoch > lowest_loss[0] + 100:
            print("No improvement for the last 100 epochs, so stopping training...")
            net.eval()
            torch.save(net, f'SuperResulutionNet_r-{r}_psnr-{int(round(np.mean(psnr) * 100))}__mse-{int(round(mean_train_loss * 10000))}')
            break




if __name__ == '__main__':
    freeze_support()
    main()
