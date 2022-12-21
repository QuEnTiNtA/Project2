import os
import shutil
from PIL import Image

from tqdm import trange

import torchvision.transforms.functional as TF
from torchvision.transforms import Resize, TenCrop


# Size of the new images: we choose the size (280,280) since its the maximum size 
#                         to take the central crop for all the considered rotation
AUG_IMG_SIZE = (280, 280)

# Angles for rotations of the originals image and mask
ANGLES = [30 * (i + 1) for i in range(11)]


def MultRotCrop(angles,size):
    def function(img):
        return [
            TF.center_crop(TF.rotate(img, angle), size)
            for angle in angles
        ]
    
    return function

def save_aug_img_mask(image: Image, mask: Image, num: int) -> None:
    """Saves an image and a mask in the corresponding file.
    Args:
        image (Image): image.
        mask (Image): mask.
        num (int): number of the image/mask.
    """
    filename = f'newImage_{num:04d}.png'
    image_path = os.path.join('data/aug_train_images', filename)
    mask_path = os.path.join('data/aug_train_masks', filename)
    image.save(image_path)
    mask.save(mask_path)


def create_augmented_dataset(replace: bool = False) -> None:
    """Creates the augmented dataset.
    It contains 2200 images of size 280x280:
    - Resize images from train data (100)
    - Crop images into four corners and the central crop plus the flipped
    version of these (100x10)
    - Rotate images and center crop (100x11)
    Args:
        replace (bool, optional): True to replace images if already exist.
        Defaults to False.
    """
    # Ignore if dataset already created
    if os.path.exists('data/aug') and not replace:
        return

    # Reset directory
    shutil.rmtree('data/aug', ignore_errors=True)

    # Creates directories
    for dirname in ('data/aug_train_masks', 'data/aug_train_images'):
        os.makedirs(dirname, exist_ok=True)

    # Get images paths
    images_names = sorted(os.listdir('data/train_images'))
    masks_names = sorted(os.listdir('data/train_masks'))

    # Define transforms
    resize = Resize(size=AUG_IMG_SIZE)
    ten_crop = TenCrop(size=AUG_IMG_SIZE)
    multiple_rotation_crop = MultRotCrop(
        angles=ANGLES,
        size=AUG_IMG_SIZE,
    )

    # Counter for the number of the image/mask
    num = 1

    # Create augmented dataset
    with trange(len(images_names), unit='image') as t:
        for i in t:
            # Retrieve images names
            image_name = images_names[i]
            mask_name = masks_names[i]

            t.set_description(desc=image_name)

            # Get images paths
            image_path = os.path.join('data/train_images', image_name)
            mask_path = os.path.join('data/train_masks', mask_name)

            # Open images
            image = Image.open(image_path)
            mask = Image.open(mask_path)

            # 1. Resize images
            image_resized = resize(image)
            mask_resized = resize(mask)

            # Save resized images
            save_aug_img_mask(image_resized, mask_resized, num)
            num += 1

            # 2. Crop images into four corners and the central crop plus the
            # flipped version of these
            images_cropped = ten_crop(image)
            mask_cropped = ten_crop(mask)

            # Save cropped images
            for i, m in zip(images_cropped, mask_cropped):
                save_aug_img_mask(i, m, num)
                num += 1

            # 3. Rotate images and center crop
            images_rotated = multiple_rotation_crop(image)
            mask_rotated = multiple_rotation_crop(mask)

            # Save rotated images
            for i, m in zip(images_rotated, mask_rotated):
                save_aug_img_mask(i, m, num)
                num += 1
