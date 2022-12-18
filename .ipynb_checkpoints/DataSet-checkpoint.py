import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms.functional as TF



class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir) 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)# grayscale
        mask = np.where(mask > 5.0, 1.0, 0.0)
        print(mask.shape)
       

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
            

        return image, mask


