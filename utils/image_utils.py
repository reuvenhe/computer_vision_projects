import time
import numpy as np
import torch
from tqdm import tqdm
import torchvision.transforms.functional as F

def image_dataset_mean_std(dataset, n_channels=3):
    # Figure out what the mean and std of the image channels are.
    # For ImageNet, the values are:  mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
    before = time.time()
    mean = torch.zeros(n_channels)
    std = torch.zeros(n_channels)
    print('==> Computing mean and std..')
    for image, _ in tqdm(dataset):
        image.unsqueeze_(0)
        for i in range(n_channels):
            mean[i] += image[:,i,:,:].mean()
            std[i] += image[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    print(f"mean/std channels values are: {mean} \n {std}")

    print("time elapsed [s]: ", time.time()-before)
    return mean, std
    ######################################################################################################################


class TransformsSquarePad:
    def __init__(self, fill_value) -> None:
        self.fill_value = fill_value

    def __call__(self, image):
        size_dims = image.size()
        max_wh = torch.max(torch.tensor([size_dims[-1], size_dims[-2]]))
        hp = int((max_wh - size_dims[-1]) / 2)
        vp = int((max_wh - size_dims[-2]) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def max_dataset_dimension(dataset):
    max_dim = 0
    for image, _ in tqdm(dataset):
        temp_max = torch.tensor(image.size()).max()
        if temp_max > max_dim:
            max_dim = temp_max

    return max_dim

