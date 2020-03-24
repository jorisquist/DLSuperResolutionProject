import math
import sys
from multiprocessing.spawn import freeze_support

import torch
from torch import optim, nn
from torchvision import transforms
import numpy as np

from SuperResolutionDataset import SuperResolutionDataset
from SuperResolutionNet import SuperResolutionNet


# Compute the PSNR using the MSE.
def mse_to_psnr(mse):
    return 10 * math.log10(1.0 / mse)


def main():
    use_gpu = torch.cuda.is_available()

    # Batch size.
    bs = 1

    # Upscale factor.
    r = 3

    # Amount of epochs.
    epochs = 20

    # Getting image data
    transform = transforms.Compose([transforms.ToTensor()])  # ,

    # Load the training data.
    training_set = SuperResolutionDataset("train_data/Set91", r, use_gpu=use_gpu)

    train_loader = torch.utils.data.DataLoader(
        training_set, batch_size=bs, shuffle=True, num_workers=0
    )

    # Initialize the network.
    net = SuperResolutionNet(r, activation=nn.ReLU())

    # Decide to use GPU or not.
    if use_gpu:
        net = net.cuda()
        print("Running on gpu")

    # Set loss function and optimizer.
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(
        net.parameters(), lr=0.1, weight_decay=1e-6, momentum=0.9, nesterov=True
    )

    # Initialize loss.
    lowest_loss = (0, float("inf"))

    # Start training.
    for epoch in range(epochs):
        train_loss = []
        psnr = []

        # Train and propagate through network.
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
        print(
            f"Epoch: {epoch: >3} Training Loss: {mean_train_loss:.6f} Mean PSNR: {np.mean(psnr):.3f}"
        )

        # Get the lowest loss.
        if mean_train_loss < lowest_loss[1]:
            lowest_loss = (epoch, mean_train_loss)

        # If we didn't improve, lets stop.
        if epoch > lowest_loss[0] + 100:
            print("No improvement for the last 100 epochs, so stopping training...")
            net.eval()
            torch.save(
                net,
                f"SuperResulutionNet_r-{r}_psnr-{int(round(np.mean(psnr) * 100))}__mse-{int(round(mean_train_loss * 10000))}",
            )
            break


if __name__ == "__main__":
    freeze_support()
    main()
