from train import *
from utils import *
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
import matplotlib.pyplot as plt
from Network import *

# Hyperparameters
lr = 1e-4
batch_size = 4
num_epochs=10
num_workers = 2
pin_memory = True
load_model = False
# load_model = True
k=5

def run_training(num_epochs, lr, batch_size, dict_double_conv, dict_ups, device=DEVICE):
    # ===== Data Loading =====
    # The input images should be normalized to have zero mean, unit variance
    # We could also add data augmentation here if we wanted

    train_dir = "data/train_images/"
    train_maskdir = "data/train_masks/"
    image_height = 600  #  400 pixels originally
    image_width = 600  #  400 pixels originally

    train_transform = get_transform(image_height, image_width, 35, 0.5, 0.1, True, 0, image_height/2)

    val_transform = get_transform(image_height, image_width, 0, 0, 0, True, 0, 0)

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

    for fold, (train_idx,val_idx) in enumerate(splits.split(train_dataset)):

        # print('Fold {}'.format(fold + 1))
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
        model = UNET(dict_double_conv, dict_ups,in_channels=3, out_channels=1,init=True).to(DEVICE)
        #model = UNET_no_skip_connection(in_channels=3, out_channels=1).to(DEVICE)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.05, patience = 20,verbose = True)

        # ===== Train Model =====
        if load_model:
             load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(1, num_epochs + 1):
            train_epoch(train_loader, model, optimizer, epoch, loss_fn, scaler, scheduler)
            
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)
            
            check_F1_score(val_loader, model, epoch)

    save_predictions_as_imgs(
    get_test_loader(batch_size, 0, True), model, folder="saved_images", device=DEVICE
)

if __name__ == "__main__":

    dict_double_conv = {"BatchNorm": True,
        "activation": nn.ReLU(inplace=True),
        "p_dropout": 0.2,
        "use_dropout": False,
        "bias": False}

    dict_ups = {"BatchNorm": False,
            "p_dropout": 0.2,
            "use_dropout": False,
            "bias": False}

    run_training(num_epochs, lr, batch_size, dict_double_conv, dict_ups, device=DEVICE) 