from skimage import io
from os import listdir
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import PIL


class SuperResolutionDataset(Dataset):

    def __init__(self, root_dir, upscale_factor, use_gpu=False):
        self.root_dir = root_dir
        self.upscale_factor = upscale_factor
        self.images = [f for f in listdir(self.root_dir) if f.endswith('.bmp') or f.endswith('.jpg')]
        self.data = list()
        for image_name in self.images:
            self.data.append(self.get_data_from_image(image_name))

        if use_gpu:
            for i in range(len(self.data)):
                self.data[i] = (self.data[i][0].cuda(), self.data[i][1].cuda())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.data[item]

    def get_data_from_image(self, image_name):
        image = io.imread(self.root_dir + '/' + image_name)

        # h, w = len(image), len(image[0])
        h, w = 256, 256
        cropped_h = h - (h % self.upscale_factor)
        cropped_w = w - (w % self.upscale_factor)


        target_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop([cropped_h, cropped_w]),
            transforms.ToTensor(),
        ])

        input_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop([cropped_h, cropped_w]),
            transforms.Resize([int(cropped_h / self.upscale_factor),
                               int(cropped_w / self.upscale_factor)],
                              PIL.Image.BICUBIC),
            transforms.ToTensor(),
        ])

        target_image = target_transform(image)
        input_image = input_transform(image)

        return input_image, target_image

    def imshow_input(self, idx):
        img, _ = self.__getitem__(idx)
        img = torchvision.utils.make_grid(img)
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def imshow_target(self, idx):
        _, img = self.__getitem__(idx)
        img = torchvision.utils.make_grid(img)
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
