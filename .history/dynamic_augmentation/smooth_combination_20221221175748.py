from model import UNET
from utils import load_checkpoint, get_test_loader
import torch
%load_ext autoreload
%autoreload 2

DEVICE = 'cuda:1'
model = UNET(in_channels=3, out_channels=1).to(DEVICE)
load_checkpoint(torch.load("checkpoint_wbce_hard_cosLR5e-4_ELU.pth", map_location='cpu'), model)
model.eval()

loader = get_test_loader(batch_size=1, num_workers=1, pin_memory=False)