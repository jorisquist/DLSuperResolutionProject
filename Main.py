import sys
from multiprocessing.spawn import freeze_support

import torch
from torch import optim, nn
from torchvision import transforms
import numpy as np

from SuperResolutionDataset import SuperResolutionDataset
from SuperResolutionNet import SuperResolutionNet


def main():
    use_gpu = torch.cuda.is_available()
    bs = 1
    r = 4

    # Getting image data
    transform = transforms.Compose(
        [transforms.ToTensor()])  # ,

    training_set = SuperResolutionDataset('train_data/Set91', r, use_gpu=use_gpu)

    train_loader = torch.utils.data.DataLoader(training_set,
                                               batch_size=bs,
                                               shuffle=True,
                                               num_workers=0)

    net = SuperResolutionNet(4, activation=nn.Tanh())
    if use_gpu:
        net = net.cuda()
        print('Running on gpu')

    loss_function = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-6, momentum=0.9, nesterov=True)

    lowest_loss = (0, float('inf'))
    for epoch in range(1000):
        train_loss = []

        net.train()
        for input, target in train_loader:
            if use_gpu:
                input = input.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            output = net(input)

            loss = loss_function(output, target)

            loss.backward()

            optimizer.step()

            train_loss.append(loss.item())

        mean_train_loss = np.mean(train_loss)
        print("Epoch:", epoch, "Training Loss: ", mean_train_loss)

        if mean_train_loss < lowest_loss[1]:
            lowest_loss = (epoch, mean_train_loss)

        if epoch > lowest_loss[0] + 10:
            print("No improvement for the last 100 epochs, so stopping training...")
            net.eval()
            break

    torch.save(net, 'saved_net')

if __name__ == '__main__':
    freeze_support()
    main()
