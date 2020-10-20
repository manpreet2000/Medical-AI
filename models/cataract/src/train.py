# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import gc
import os
import cv2
from sklearn.metrics import precision_score, recall_score,f1_score, confusion_matrix
from torch.utils.data import DataLoader,SubsetRandomSampler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torchvision import transforms,models
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# import custom packages
import config
import preprocess
import dataset_class

device='cuda:0' if torch.cuda.is_available() else 'cpu'


train_df=preprocess.preprocess_me(config.oc_path)

train_df,test_df=train_test_split(train_df,test_size=0.12,shuffle=True,stratify=train_df.cataract)


transform=transforms.Compose([
     transforms.ToPILImage(),
    transforms.Resize((config.IMG_SIZE,config.IMG_SIZE)),
     transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set=dataset_class.cat_dataset(train_df,transforms=transform)
test_set=dataset_class.cat_dataset(test_df,transforms=transform)


train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.BATCH,shuffle=True)
val_loader = torch.utils.data.DataLoader(test_set, batch_size=config.BATCH,shuffle=True)


model=models.densenet121(pretrained=True)
model.classifier=nn.Sequential(nn.Linear(1024,2))
model=model.to(device)

crit=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters())
sch=ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=0, verbose=True)

def train(model, epochs, optimizer, train_loader, criterion,test_loader,sch=None):
    for epoch in range(1,epochs+1):
        # train
        total_loss = 0
        total=0
        model.train()
        correct=0
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            # acc+=binary_acc(output.view(-1),target)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
            loss.backward()
            optimizer.step()
        acc = 100 * correct / total
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader),
            100. * batch_idx / len(train_loader), total_loss /len(train_loader)))
        #print('Train Accuracy for epoch {} is {} \n'.format(epoch,100. *correct/len(train_loader.dataset)))
        print("Train acc \n",acc)
        # test
        model.eval()
        test_loss = 0
        correct=0
        best_acc=0
        total=0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
        acc = 100 * correct / total
        if best_acc<acc:
          best_acc=acc
          torch.save(model.state_dict(),config.WEIGHT)
          print("Model saved \n")

        test_loss /= len(test_loader)
        print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
        print("Test acc \n",acc)
        if sch:
              sch.step(acc)
        

if os.path.exists(config.WEIGHT):
    print("\n Model found! Loading \n")
    
    if torch.cuda.is_available()==False:
        model.load_state_dict(torch.load(config.WEIGHT, map_location=lambda storage, loc: storage))
    else:
        model.load_state_dict(torch.load(config.WEIGHT))
else:
    train(model, EPOCHS, optimizer, train_loader, crit,val_loader)



model.eval()
test_loss = 0
correct=0
total=0
pred=[]
lab=[]
with torch.no_grad():
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += crit(output, target).item()
        _, predicted = torch.max(output.data, 1)
        pred.extend(predicted.cpu().numpy().tolist())
        lab.extend(target.cpu().numpy().tolist())
        correct += (predicted == target).sum().item()
        total += target.size(0)
acc = 100 * correct / total


test_loss /= len(val_loader)
print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
print("Test acc \n",acc)
print("precision",precision_score(lab,pred))
print("recall",recall_score(lab,pred))
print("F1 score",f1_score(lab,pred))
cm=confusion_matrix(lab,pred)
print("Sensitivity",cm[0,0]/(cm[0,0]+cm[0,1]))
print("Specifity",cm[1,1]/(cm[1,1]+cm[1,0]))
print("PPV",cm[0,0]/(cm[0,0]+cm[1,0]))
print("PNV",cm[1,1]/(cm[1,1]+cm[0,1]))