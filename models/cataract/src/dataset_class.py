import torch
import pandas as pd
import numpy as np
import cv2
from torchvision import transforms

class cat_dataset(torch.utils.data.Dataset):
    def __init__(self,df,transforms=None):
        self.df=df
        self.transforms=transforms
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        
        img=cv2.imread(self.df.Path.iloc[idx])
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if self.transforms:
            img=self.transforms(img)
        label=self.df.cataract.iloc[idx]
        return (img,label)