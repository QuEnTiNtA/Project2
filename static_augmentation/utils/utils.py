import os
import shutil
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from train import *

"""
Save a simulation state into a checkpoint
"""
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


"""
Load a simulation checkpoint
"""
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

"""
Definition of the dataset
"""
class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir) # list all the files that are in that folder

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) # grayscale
        mask = np.where(mask > 4.0, 1.0, 0.0)
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

"""
Returns the set of transformations that will be performed on the dataset
depending on the parameters given
"""
def get_transform(
    image_height,
    image_width,
    max_rotation,
    p_hflip,
    p_vflip,
    normalize,
    crop_width,
    crop_height
):
    cwidth = int(crop_width)
    cheight = int(crop_height)
    transformations = []
    if(image_height != 0):
        transformations.append(A.Resize(height=image_height, width=image_width))
    if(max_rotation > 0):
        transformations.append(A.Rotate(limit=max_rotation, p=1.0))
    if(p_hflip > 0):
        transformations.append(A.HorizontalFlip(p=p_hflip))
    if(p_vflip > 0):
        transformations.append(A.VerticalFlip(p=p_vflip))
    if(normalize):
        transformations.append(A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ))
    if(cwidth > 0):
        transformations.append(A.RandomCrop(width=cwidth, height=cheight))
    transformations.append(ToTensorV2())
    return A.Compose(transformations)


test_dir = '../data/test_images/'

path_list = os.listdir(test_dir)
path_list.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
path_list


"""
Defines the test set of the dataset
"""
class RoadData_test_set(Dataset):

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = path_list # list all the files that are in that folder

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image


"""
Returns the loader for the dataset
"""
def get_test_loader(num_workers, pin_memory):

    image_height = 600  #  400 pixels originally
    image_width = 600  #  400 pixels originally

    test_transform = get_transform(image_height, image_width, 0, 0, 0, True, 0, 0)

    test_dataset = RoadData_test_set(
        image_dir = test_dir,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return test_loader
