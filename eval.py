import PIL
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms

from SuperResolutionDataset import SuperResolutionDataset
import numpy as np


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


net = torch.load('saved_net')
r = net.r

use_gpu = torch.cuda.is_available()
training_set = SuperResolutionDataset('train_data/Set91', r, use_gpu=use_gpu)

train_loader = torch.utils.data.DataLoader(training_set,
                                           batch_size=1,
                                           shuffle=True,
                                           num_workers=0)

input, target = iter(train_loader).next()

if use_gpu:
        input = input.cuda()
        target = target.cuda()

output = net(input)

if use_gpu:
  input = input.cpu()
  output = output.cpu()
  target = target.cpu()

print(input[0].size())
bicubic = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([int(r * input.size()[2]),
                               int(r * input.size()[3])],
                               PIL.Image.BICUBIC),
            transforms.ToTensor()
       ])
bicubic_upscaled = bicubic(input[0])

imshow(torchvision.utils.make_grid(input))
imshow(torchvision.utils.make_grid(target))
imshow(torchvision.utils.make_grid(output.detach()))
imshow(torchvision.utils.make_grid(bicubic_upscaled))