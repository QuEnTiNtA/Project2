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

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, params_DoubleConv, init = False):
        super(DoubleConv, self).__init__()
        
        ordered_dict = OrderedDict([("conv1",nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=params_DoubleConv["bias"]))])
        
        if params_DoubleConv["BatchNorm"]:
            ordered_dict["BatchNorm1"] = nn.BatchNorm2d(out_channels)
        ordered_dict["activation1"] = params_DoubleConv["activation"]
        if params_DoubleConv["use_dropout"] != None:
            ordered_dict["Dropout1"] = nn.Dropout(p=params_DoubleConv["p_dropout"])
            
        ordered_dict["conv2"] = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=params_DoubleConv["bias"])
        if params_DoubleConv["BatchNorm"]:
            ordered_dict["BatchNorm2"] = nn.BatchNorm2d(out_channels)
        ordered_dict["activation2"] = params_DoubleConv["activation"]
        if params_DoubleConv["p_dropout"] != None:
            ordered_dict["Dropout2"] = nn.Dropout(p=params_DoubleConv["p_dropout"])
            
        if init:
            nn.init.xavier_normal_(ordered_dict["conv1"].weight)
            nn.init.xavier_normal_(ordered_dict["conv2"].weight)
            
            if params_DoubleConv["bias"]:
                ordered_dict["conv1"].bias.data.fill_(0.01)
                ordered_dict["conv2"].bias.data.fill_(0.01)
        
        self.conv = nn.Sequential(ordered_dict)
        

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, params_DoubleConv, params_up ,in_channels=3, out_channels=1, features=[64, 128, 256, 512],init = False,
    ): # outchannels = 1: binary class
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature,params_DoubleConv,init))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            ordered_dict = OrderedDict([])
            if params_up["BatchNorm"]:
                ordered_dict["BatchNorm"] = nn.BatchNorm2d(feature*2)
                
            ordered_dict["trans_conv"] = nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2,)
            
            if init:    
                nn.init.xavier_normal_(ordered_dict["trans_conv"].weight)
                if params_DoubleConv["bias"]:
                    ordered_dict["trans_conv"].bias.data.fill_(0.01)
                    
            if params_up["use_dropout"]:
                ordered_dict["Dropout"] = nn.Dropout(p=params_up["p_dropout"])
                
            self.ups.append(nn.Sequential(ordered_dict))
            
            self.ups.append(DoubleConv(feature*2, feature,params_DoubleConv,init))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2,params_DoubleConv,init)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        if init:
            nn.init.xavier_normal_(self.final_conv.weight)
            if params_DoubleConv["bias"]:
                self.final_conv.bias.data.fill_(0.01)
            

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = nn.BatchNorm2d(x.shape[1])(x)
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:]) # height and width

            concat_skip = torch.cat((skip_connection, x), dim=1) # concatenate along in-channel axis
            
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    
class UNET_no_skip_connection(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ): # outchannels = 1: binary class
        super(UNET_no_skip_connection, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                ) # feature*2: skip connection
            )
            self.ups.append(DoubleConv(feature, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:]) # height and width

            #concat_skip = torch.cat((skip_connection, x), dim=1) # concatenate along in-channel axis
            x = self.ups[idx+1](x)

        return self.final_conv(x)


def train_epoch(train_loader, model, optimizer, epoch, loss_fn, scaler, scheduler):
    loop = tqdm(train_loader)

    TP = 0
    FP = 0
    TN = 0
    FN = 0
    num_correct = 0
    num_pixels = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE) # add a channel dimension
        # forward
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = loss_fn(output, targets)
            
        scheduler.step(loss)
        
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        preds = torch.sigmoid(output)
        preds = (preds > 0.5).float()
        #print(TP,FP,FN,preds.sum())
        TP += ((preds == 1)*(targets==1)).sum()
        FP += ((preds == 1)*(targets==0)).sum()
        FN += ((preds == 0)*(targets==1)).sum()
        num_pixels += torch.numel(preds)
        num_correct += (preds == targets).sum()

    recall = TP/(TP+FN)
    precision = TP/(TP+FP)

    print(
        f"Training set: epoch-{epoch} got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}% and F1-score {2*recall*precision/(recall+precision):.2f} "
    )

    return num_correct/num_pixels, 2*recall*precision/(recall+precision)


def check_F1_score(val_loader, model, epoch):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device=DEVICE)
            y = y.to(device=DEVICE).unsqueeze(1) # the grayscale does not have channels, add
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            #print(TP,FP,FN,preds.sum())
            TP += ((preds == 1)*(y==1)).sum()
            FP += ((preds == 1)*(y==0)).sum()
            FN += ((preds == 0)*(y==1)).sum()
            num_pixels += torch.numel(preds)
            num_correct += (preds == y).sum()
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    
    print(
        f"Validation set: epoch-{epoch} got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}% and F1-score {2*recall*precision/(recall+precision):.2f}"
    )
    model.train()

    return num_correct/num_pixels, 2*recall*precision/(recall+precision)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_training(num_epochs, lr, batch_size, dict_double_conv, dict_ups, device=DEVICE):
    # ===== Data Loading =====
    # The input images should be normalized to have zero mean, unit variance
    # We could also add data augmentation here if we wanted

    train_dir = "data/train_images/"
    train_maskdir = "data/train_masks/"
    image_height = 400  #  400 pixels originally
    image_width = 400  #  400 pixels originally

    train_transform = get_transform(image_height, image_width, 35, 0.5, 0.1, True, image_width/2, image_height/2)

    val_transform = get_transform(image_height, image_width, 0, 0, 0, True, image_width/2, image_height/2)

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
                                  pin_memory=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                                sampler=val_sampler,
                                pin_memory=True, num_workers=0)        

        # ===== Model, Optimizer and Loss function =====   
        model = UNET(dict_double_conv, dict_ups,in_channels=3, out_channels=1,init=True).to(DEVICE)
        #model = UNET_no_skip_connection(in_channels=3, out_channels=1).to(DEVICE)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.05, patience = 20,verbose = True)

        # ===== Train Model =====
        for epoch in range(1, num_epochs + 1):
            train_epoch(train_loader, model, optimizer, epoch, loss_fn, scaler, scheduler)
            check_F1_score(val_loader, model, epoch)
  

lr = 1e-4
batch_size = 4
num_epochs=10
k=5

dict_double_conv = {"BatchNorm": True,
        "activation": nn.ReLU(inplace=True),
        "p_dropout": 0.2,
        "use_dropout": False,
        "bias": False}

dict_ups = {"BatchNorm": False,
        "p_dropout": 0.2,
        "use_dropout": False,
        "bias": False}

scaler = torch.cuda.amp.GradScaler()
run_training(num_epochs, lr, batch_size, dict_double_conv, dict_ups, device=DEVICE)  