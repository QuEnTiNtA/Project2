import os
import shutil
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from train import *

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

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
        # mask[mask == 237.0] = 1.0 # white --> 1, implement sigmoid as the last activation. rgb(0,0,0): black, rgb(255, 255, 255):

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

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
            max_pixel_value=237.0, # dividing by 237, get a value between 0 and 1
        ))
    if(cwidth > 0):
        transformations.append(A.RandomCrop(width=cwidth, height=cheight))
    transformations.append(ToTensorV2())
    return A.Compose(transformations)

# if fold 'test_images' is empty
# determination = 'data/test_images/'
# if not os.path.exists(determination):
#     os.makedirs(determination)

# path = 'data/test/'
# folders = os.listdir(path)
# for folder in folders:
#     dir = path + '/' + str(folder)
#     files = os.listdir(dir)
#     for file in files:
#         source = dir + '/' + str(file)
#         deter = determination + '/' + str(file)
#         shutil.copyfile(source, deter)

test_dir = 'data/test_images/'

path_list = os.listdir(test_dir)
path_list.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
path_list

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

def get_test_loader( num_workers, pin_memory):

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