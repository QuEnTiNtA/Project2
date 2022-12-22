import numpy as np
from PIL import Image
import torchvision
from model import UNET
from utils import load_checkpoint, get_test_loader, combine_patches
import torch

DEVICE = 'cuda:0'
model = UNET(in_channels=3, out_channels=1).to(DEVICE)
load_checkpoint(torch.load("checkpoint_wbce_hard_StepLR1e-4_ELU.pth", map_location='cpu'), model)
model.eval()

loader = get_test_loader(batch_size=1, num_workers=1, pin_memory=False)

# produce results
n_patch_x = 5
n_patch_y = 5
step_x = (608 - 304) // (n_patch_x - 1)
step_y = (608 - 304) // (n_patch_y - 1)
for idx, x in enumerate(loader):
    x = x[0].to(device=DEVICE)
    with torch.no_grad():
        if x.shape[0] < 50:
            preds = torch.sigmoid(model(x))
        else:  # if the batch size is too large for memory
            preds_1 = torch.sigmoid(model(x[:50]))
            preds_2 = torch.sigmoid(model(x[50:]))
            preds = torch.cat([preds_1, preds_2])
    preds = preds.reshape(n_patch_x, n_patch_y, 1, 304, 304).cpu()
    pred_img = combine_patches(preds, n_patch_x=n_patch_x, n_patch_y=n_patch_y, step_x=step_x, step_y=step_y, solve_overlap='mean')
    pred_img = (pred_img > 0.5).float()
    torchvision.utils.save_image(
        torch.Tensor(pred_img), f"saved_images_test/pred_{idx+1}.png"
    )