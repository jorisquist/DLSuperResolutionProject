import math

import PIL
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torchvision import transforms
from os import listdir

from SuperResolutionDataset import SuperResolutionDataset
from SuperResolutionTestSet import SuperResolutionTestSet
import numpy as np


def imshow(img):
    npimg = img.numpy()
    plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def evaluate(netork_name):
    net = torch.load(netork_name)
    r = net.r
    print(f"r: {r}")

    use_gpu = torch.cuda.is_available()

    # test_set_paths = ["test_data/" + f for f in listdir("test_data")]
    test_set_paths = ["test_data/Custom"]

    for path in test_set_paths:
        psnr = []
        test_set = SuperResolutionTestSet(path, r, use_gpu=use_gpu)

        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=1,
                                                  shuffle=True,
                                                  num_workers=0)

        for input, target in iter(test_loader):
            if use_gpu:
                input = input.cuda()
                target = target.cuda()

            if input.size()[1] == 1:
                input = input.repeat(1, 3, 1, 1)


            print(f"###  input dimensions: {input.size()}")

            output = net(input)


            if use_gpu:
                input = input.cpu()
                output = output.cpu()
                target = target.cpu()


            bicubic = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([int(r * input.size()[2]),
                                   int(r * input.size()[3])],
                                  PIL.Image.BICUBIC),
                transforms.ToTensor()
            ])
            bicubic_upscaled = bicubic(input[0])

            nearest_neighbour = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([int(r * input.size()[2]),
                                   int(r * input.size()[3])],
                                  PIL.Image.NEAREST),
                transforms.ToTensor()
            ])

            input = nearest_neighbour(input[0])
            output = torch.clamp(output.detach(), 0, 1)

            if target.size()[1] == 1:
                target = target.repeat(1, 3, 1, 1)

            mse_loss = nn.MSELoss()
            psnr.append(10 * math.log10(1. / mse_loss(output, target).item()))

            images = [input, target[0], output.detach()[0], bicubic_upscaled]

            imshow(torchvision.utils.make_grid(images))
            torchvision.utils.save_image(output, "test-image-out.png")
            torchvision.utils.save_image(input, "test-image-in.png")

        print(f"{path} psnr: {np.mean(psnr)}")

    # plt.show()


if __name__ == '__main__':
    evaluate('SuperResulutionNet_r-3_psnr-3071__mse-8-JORIS-DESKTOP')
