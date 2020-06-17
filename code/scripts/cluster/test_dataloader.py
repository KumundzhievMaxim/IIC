# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

import os
from skimage import io

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5, ))
])


class SampleDataset(Dataset):
    """Create Sample dataset.
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): path to folder with images;
            transform: applied transformations for images;
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = [name.split('/')[-1] for name in os.listdir(root_dir)]

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        image_path = os.path.join(self.root_dir,
                                  self.image_names[idx])
        image = io.imread(image_path)

        if self.transform:
            image = self.transform(image)

        return image


dataset = SampleDataset(
    root_dir='/Users/macbook/Desktop/IIC/IIC/data/dataset-single/blowjob',
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=4,
                        shuffle=True, num_workers=4)



