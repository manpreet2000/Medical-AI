import numpy as np
import pandas as pd
import os
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensor, ToTensorV2
from sklearn.model_selection import train_test_split
import cv2
from torch.optim.lr_scheduler import StepLR
# custom packages
import config
import unet_arch
import data_load
import dataset_class
import plot_everything
import dice_loss

print("Training ")
df=data_load.load_images_in_df()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


transforms = A.Compose([
    A.Resize(width = config.IMAGE_SIZE, height = config.IMAGE_SIZE, p=1.0),
    A.HorizontalFlip(p=0.2),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.2),
    A.Transpose(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
    A.Normalize(p=1.0),
    ToTensor(),
])

print("Spliting data")
# Split df into train_df and val_df
train_df, val_df = train_test_split(df, stratify=df.diagnosis, test_size=0.1)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# Split train_df into train_df and test_df
train_df, test_df = train_test_split(train_df, stratify=train_df.diagnosis, test_size=0.15)
train_df = train_df.reset_index(drop=True)

#train_df = train_df[:1000]
print(f"Train: {train_df.shape} \nVal: {val_df.shape} \nTest: {test_df.shape}")

train_dataset = dataset_class.BrainMriDataset(df=train_df, transforms=transforms)
train_dataloader = DataLoader(train_dataset, batch_size=26, num_workers=4, shuffle=True)

# val
val_dataset = dataset_class.BrainMriDataset(df=val_df, transforms=transforms)
val_dataloader = DataLoader(val_dataset, batch_size=26, num_workers=4, shuffle=True)

#test
test_dataset = dataset_class.BrainMriDataset(df=test_df, transforms=transforms)
test_dataloader = DataLoader(test_dataset, batch_size=26, num_workers=4, shuffle=True)


    
images, masks = next(iter(train_dataloader))
plot_everything.show_aug(images)
plot_everything.show_aug(masks, image=False)

model=unet_arch.UNet()

def train(model,optimizer,epoch,lr_schedular,train_loader):
    model.train()
    loss_collection=0
    total_size=0
    for i,(img,lab) in enumerate(train_loader):
        optimizer.zero_grad()
        img,lab=img.to(device),lab.to(device)
        pred=model(img)
        loss=dice_loss.bce_dice_loss(pred,lab)
        loss_collection+=loss.item()
        total_size += img.size(0)
        loss.backward()
        optimizer.step()
        if i%50==0:
            lr_schedular.step()
            print("\n Learning rate is {}".format(lr_schedular.get_last_lr()))
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                epoch, i * len(img), len(train_loader.dataset),
                100. * i / len(train_loader), loss_collection / total_size))
    return loss_collection

best_loss=10

def test(model,test_loader,tst=None):
    global best_loss
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += dice_loss.bce_dice_loss(output, target).item()
    test_loss /= len(test_loader.dataset)
    if(test_loss<best_loss and tst==None):
        print('\n MODEL SAVED!! loss decreased from {} to {} \n'.format(best_loss,test_loss))
        best_loss=test_loss
        #torch.save(model.state_dict(),'./models/brain tumor/weights/model.h5')
        
    
    print('\nTest set: Average loss: {:.4f}'.format(
        test_loss,))
    return test_loss


if os.path.exists(config.WEIGHT):
    print("\n Model found! Loading \n")
    
    if torch.cuda.is_available()==False:
        model.load_state_dict(torch.load(config.WEIGHT, map_location=lambda storage, loc: storage))
    else:
        model.load_state_dict(torch.load(config.WEIGHT))

else:
    def start_train(model,epochs,optimizer,train_loader,test_loader,lr_sch):
        model=model.to(device)
        train_loss,test_loss=[],[]
        for epoch in range(1,epochs+1):
            train_loss.append(train(model,optimizer,epoch,lr_sch,train_loader))
            test_loss.append(test(model,test_loader))
        return train_loss,test_loss

    optimizer=torch.optim.Adam(model.parameters(),lr=config.LR)
    lr_sch=StepLR(optimizer, step_size=config.STEP_SIZE, gamma=0.96)
    train_loss,test_loss=start_train(model,config.EPOCHS,optimizer,train_dataloader,val_dataloader,lr_sch)

print("*"*100)
print(" TEST LOSS \n")
_=test(model,test_dataloader,tst="test")