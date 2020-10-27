import torch
import numpy as np
from torchvision import models,transforms
import torch.nn as nn
import cv2


device='cuda:0' if torch.cuda.is_available() else 'cpu'

model_pre = models.densenet121()
for param in model_pre.features.parameters():
    param.required_grad = False

num_features = model_pre.classifier.in_features
features = list(model_pre.classifier.children())[:-1] 
features.extend([nn.Linear(num_features, 2)])
model_pre.classifier = nn.Sequential(*features)
model_pre = model_pre.to(device)

if torch.cuda.is_available()==False:
    model.load_state_dict(torch.load("models/pneumonia/weight/pne.pt", map_location=lambda storage, loc: storage))
else:
    model.load_state_dict(torch.load("models/cataractpneumonia/weight/pne.pt"))

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

class predict_img(object):
    def __init__(self,image_path):
        self.image_path=image_path

    def predict_cataract(self):
        img=cv2.imread(self.image_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=transform(img)
        try:
            img=img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
        except:
            img=img.reshape((1,img.shape[0],img.shape[1]))
        model.eval()
        with torch.no_grad():
            pred=model(img.to(device))
        _,predicted = torch.max(pred.data, 1)

        predicted="Pneumonia" if predicted==1 else "Normal"
        return predicted