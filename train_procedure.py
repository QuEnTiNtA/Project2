import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torchvision.transforms.functional as TF
from tqdm import tqdm
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from collections import OrderedDict
import matplotlib.pyplot as plt
from Network import *
from DataSet import *
from torchvision import transforms

def train_epoch(train_loader, model, optimizer, epoch, loss_fn, scaler,device = 'cpu', scheduler=None):
    loop = tqdm(train_loader)

    TP = 0
    FP = 0
    TN = 0
    FN = 0
    num_correct = 0
    num_pixels = 0
    
    loss_list = []

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().to(device=device) # add a channel dimension
        # forward
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = loss_fn(output, targets)
            loss_list.append(loss)
            
        if scheduler != None:
            scheduler.step()
        
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        preds = torch.sigmoid(output)
        preds = (preds > 0.5).float()
        
        TP += ((preds == 1)*(targets==1)).sum()
        FP += ((preds == 1)*(targets==0)).sum()
        FN += ((preds == 0)*(targets==1)).sum()
        num_pixels += torch.numel(preds)
        num_correct += (preds == targets).sum()
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)

    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    print(
        f"Training set: epoch-{epoch} got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}%, F1-score {2*recall*precision/(recall+precision):.2f} and loss {sum(loss_list)/len(loss_list)} ")

    return num_correct/num_pixels, 2*recall*precision/(recall+precision), loss_list


def check_F1_score(val_loader, model, epoch, loss_fn,device='cpu'):
    TP = 0
    FP = 0
    FN = 0
    num_correct = 0
    num_pixels = 0
    model.eval()
    loss_ = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device=device)
            y = y.to(device=device) # the grayscale does not have channels, add
            output = model(x)
            loss_.append(loss_fn(output, y))
            preds = torch.sigmoid(output)
            preds = (preds > 0.5).float()
            
            TP += ((preds == 1)*(y==1)).sum()
            FP += ((preds == 1)*(y==0)).sum()
            FN += ((preds == 0)*(y==1)).sum()
            num_pixels += torch.numel(preds)
            num_correct += (preds == y).sum()
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    
    print(
        f"Validation set: epoch-{epoch} got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}% , F1-score {2*recall*precision/(recall+precision):.2f} and validation loss {sum(loss_)/len(loss_)}"
    )
    model.train()

    return num_correct/num_pixels, 2*recall*precision/(recall+precision), loss_




def run_training(dict_training):
    
    convergence_path = {}
    
    train_dir = 'data/aug_train_images/'#"data/train_images/"
    train_maskdir = 'data/aug_train_masks/'#"data/train_masks/"
    image_height = 400  #  400 pixels originally
    image_width = 400  #  400 pixels originally
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = RoadDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )
    
    if dict_training["cross_val"]:
        splits=KFold(n_splits=dict_training["n_splits"],shuffle=True,random_state=42)

        for fold, (train_idx,val_idx) in enumerate(splits.split(train_dataset)):

            print(f'-------------------------- {fold + 1} fold --------------------------')

            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            train_loader = DataLoader(train_dataset, batch_size=dict_training["batch_size"], 
                                      sampler=train_sampler, 
                                      pin_memory=True, num_workers=0)
            val_loader = DataLoader(train_dataset, batch_size=dict_training["batch_size"], 
                                    sampler=val_sampler,
                                    pin_memory=True, num_workers=0)        

            # ===== Model, Optimizer and Loss function =====   

            if dict_training["skip_connection"]:
                model = UNET(dict_training["dict_double_conv"], dict_training["dict_ups"],in_channels=3, out_channels=1,init=True,
                             scale_channel=dict_training["scale_channel"]).to(dict_training["device"])
            else:
                model = UNET_no_skip_connection(dict_training["dict_double_conv"],in_channels=3, out_channels=1,
                                                init=True,scale_channel=dict_training["scale_channel"]).to(dict_training["device"])
            
            loss_fn = dict_training["loss"]

            if dict_training["param_optimizer"]["weight_decay"] != None:
                optimizer = dict_training["optimizer"](model.parameters(), lr = dict_training["param_optimizer"]["lr"], weight_decay = dict_training["param_optimizer"]["weight_decay"])
            else:
                optimizer = dict_training["optimizer"](model.parameters(), lr = dict_training["param_optimizer"]["lr"])

            if dict_training["use_scheduler"]:
                if dict_training["type_scheduler"] == "StepLR":
                    scheduler = dict_training["scheduler"](optimizer,step_size=dict_training["param_scheduler"]["step_size"], gamma=dict_training["param_scheduler"]["gamma"])
            
            fold_path = {"train_acc": [],
                        "train_F1": [],
                        "train_loss": [],
                        "val_acc": [],
                        "val_F1": [],
                        "val_loss": []}
            if dict_training["save_model"]:
                fold_path["last_model"] = model.state_dict()
                fold_path["best_model"] = model.state_dict()
            # ===== Train Model =====
            
            for epoch in range(1, dict_training["num_epochs"] + 1):
                
                train_acc, train_F1, train_loss  = train_epoch(train_loader, model, optimizer, epoch, loss_fn, dict_training["scaler"],device=dict_training["device"])
                val_acc, val_F1, val_loss = check_F1_score(val_loader, model, epoch, loss_fn,device=dict_training["device"])
                
                fold_path["train_acc"].append(train_acc)
                fold_path["train_F1"].append(train_F1)
                fold_path["train_loss"].append(train_loss)
                fold_path["val_acc"].append(val_acc)
                fold_path["val_F1"].append(val_F1)
                fold_path["val_loss"].append(val_loss)
                
                if dict_training["save_model"] and epoch>1 and val_F1 >= max(fold_path["val_F1"][:-1]):
                    fold_path["best_model"] = model.state_dict()
                    
                    
                if dict_training["use_scheduler"]:
                    if dict_training["type_scheduler"] == "StepLR":
                        scheduler.step()
            
            if dict_training["save_model"]:
                fold_path["last_model"] = model.state_dict()
            convergence_path[f"Kfold {fold}"] = fold_path
    else:
        train_loader = DataLoader(train_dataset, batch_size=dict_training["batch_size"], 
                                      pin_memory=True, num_workers=0)      

            # ===== Model, Optimizer and Loss function =====   

        if dict_training["skip_connection"]:
            model = UNET(dict_training["dict_double_conv"], dict_training["dict_ups"],in_channels=3, out_channels=1,init=True,scale_channel=dict_training["scale_channel"]).to(dict_training["device"])
        else:
            model = UNET_no_skip_connection(dict_training["dict_double_conv"],in_channels=3, out_channels=1,init=True,scale_channel=dict_training["scale_channel"]).to(dict_training["device"])

        loss_fn = dict_training["loss"]

        if dict_training["param_optimizer"]["weight_decay"] != None:
            optimizer = dict_training["optimizer"](model.parameters(), lr = dict_training["param_optimizer"]["lr"], weight_decay = dict_training["param_optimizer"]["weight_decay"])
        else:
            optimizer = dict_training["optimizer"](model.parameters(), lr = dict_training["param_optimizer"]["lr"])

        if dict_training["use_scheduler"]:
            if dict_training["type_scheduler"] == "StepLR":
                scheduler = dict_training["scheduler"](optimizer,step_size=dict_training["param_scheduler"]["step_size"], gamma=dict_training["param_scheduler"]["gamma"])

        convergence_path = {"train_acc": [],
                        "train_F1": [],
                        "train_loss": [],
                        "val_acc": [],
                        "val_F1": [],
                        "last_model": model.state_dict(),
                        "best_model": model.state_dict() }
        
            # ===== Train Model =====

        for epoch in range(1, dict_training["nums_epochs"] + 1):
            
            train_acc, train_F1, train_loss  = train_epoch(train_loader, model, optimizer, epoch, loss_fn, dict_training["scaler"], device=dict_training["device"])
            
            convergence_path["train_acc"].append(train_acc)
            convergence_path["train_F1"].append(train_F1)
            convergence_path["train_loss"].append(train_loss)
                
            if val_F1 >= max(fols_path["val_F1"]):
                convergence_path["best_model"] = model.state_dict()
                
            if dict_training["use_scheduler"]:
                if dict_training["type_scheduler"] == "StepLR":
                    scheduler.step()
                
        convergence_path["last_model"] = model.state_dict()
            
    return convergence_path
