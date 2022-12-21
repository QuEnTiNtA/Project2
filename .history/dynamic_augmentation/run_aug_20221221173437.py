from train import *
from utils import *
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
import matplotlib.pyplot as plt
from model import *

import wandb

# Hyperparameters
lr = 5e-4
# lr = 1e-4
batch_size = 20
num_epochs = 1000
num_workers = 2
pin_memory = True
load_model = False
# load_model = True
k = 5
checkpoint_name = 'checkpoint_cosLR5e-4_ELU_1000epoch.pth'

def run_training(num_epochs, lr, batch_size, device=DEVICE):
    # ===== Data Loading =====
    # The input images should be normalized to have zero mean, unit variance
    # We could also add data augmentation here if we wanted

    train_dir = "data/train_images"
    train_maskdir = "data/train_masks"

    train_transform = get_transform(train=True)
    val_transform = get_transform(train=False)

    train_dataset = RoadDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    val_dataset = RoadDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=val_transform,
    )
    

    splits=KFold(n_splits=5,shuffle=True,random_state=42)

    for fold, (train_idx,val_idx) in enumerate(splits.split(dataset)):

        print('--------------------------', fold + 1, 'fold --------------------------')

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                  sampler=train_sampler, 
                                  pin_memory=pin_memory, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                                sampler=val_sampler,
                                pin_memory=pin_memory, num_workers=num_workers)        

        # ===== Model, Optimizer and Loss function =====   
        model = UNET(in_channels=3, out_channels=1).to(DEVICE)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.05, patience = 20,verbose = True)

        # ===== Train Model =====
        if load_model:
             load_checkpoint(torch.load("my_checkpoint.pth"), model)

        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(1, num_epochs + 1):
            train_epoch(train_loader, model, optimizer, epoch, loss_fn, scaler)
            
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)
            
            check_F1_score(val_loader, model, epoch)

    save_predictions_as_imgs(
        get_test_loader(batch_size=1, num_workers = 2, pin_memory = True), model, folder="saved_images_test", device=DEVICE
    )

    save_val_predictions_as_imgs(
        val_loader, model, folder="saved_images_val", device=DEVICE
    )


def run_training_without_CV(num_epochs, lr, batch_size, device=DEVICE, weighted_sampler=False):
    # ===== Data Loading =====
    # The input images should be normalized to have zero mean, unit variance
    # We could also add data augmentation here if we wanted
    train_dir = "data/train_images"
    train_maskdir = "data/train_masks"
    test_dir = 'data/test_images/'

    transform = get_transform(train=True)

    dataset = RoadDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=transform,
    )

    if weighted_sampler == True:
        hard_examples_idx = [10, 18, 20, 22, 26, 27, 28, 29, 30, 31, 32, 40, 52, 74, 76, 77, 85, 87, 90, 91]
        weights = [3.0 if idx in hard_examples_idx else 1.0 for idx in range(len(dataset))]
        sampler = WeightedRandomSampler(weights, len(dataset))
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                  pin_memory=pin_memory, shuffle=False, num_workers=num_workers)     
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                  pin_memory=pin_memory, shuffle=True, num_workers=num_workers)     

    # ===== Model, Optimizer and Loss function =====   
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    # loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([4]).to(DEVICE))  # positive : negative = 1 : 4
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler()

    wandb.config.update({'optimizer': optimizer,
                         'scheduler': scheduler,
                         'transforms': transform})

    # ===== Train Model =====
    if load_model:
        load_checkpoint(torch.load(checkpoint_name, map_location='cpu'), model)

    best_f1 = 0
    for epoch in range(1, num_epochs + 1):
        acc, f1 = train_epoch(train_loader, model, optimizer, epoch, loss_fn, scaler)
        if f1 > best_f1:
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=checkpoint_name)
            best_f1 = f1

    # Reload the best checkpoint
    load_checkpoint(torch.load(checkpoint_name), model)
    wandb.save(checkpoint_name)

    save_predictions_as_imgs(
        get_test_loader(batch_size=1, num_workers=2, pin_memory=True), model, folder="saved_images_test", device=DEVICE
    )


if __name__ == "__main__":
    wandb.init(project='sadist')

    CROSS_VALIDATION = False
    if CROSS_VALIDATION:
        run_training(num_epochs, lr, batch_size, device=DEVICE) 
    else:
        # run_training_without_CV(num_epochs, lr, batch_size, device=DEVICE) 
        run_training_without_CV(num_epochs, lr, batch_size, device=DEVICE, weighted_sampler=False) 

