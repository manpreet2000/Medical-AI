# importing libraries
import torch
import numpy as np
from torchvision import models,transforms
import torch.nn as nn
import cv2
import base64
from PIL import Image
from io import BytesIO

device='cuda:0' if torch.cuda.is_available() else 'cpu'
IMG_SIZE=256

model=models.densenet121(pretrained=True)
model.classifier=nn.Sequential(nn.Linear(1024,2))
model=model.to(device)



if torch.cuda.is_available()==False:
    model.load_state_dict(torch.load("models/cataract/weight/cat1.h5", map_location=lambda storage, loc: storage))
else:
    model.load_state_dict(torch.load("models/cataract/weight/cat.h5"))

transform=transforms.Compose([
     transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
     transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



class predict_img(object):
    def __init__(self,image_code):
        self.image_code=image_code

    def predict_cataract(self):
        img=cv2.imdecode(self.image_code,cv2.IMREAD_COLOR)
        imgo = Image.fromarray(img.astype("uint8"))
        img=np.array(imgo)
        img=transform(img)
        try:
            img=img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
        except:
            img=img.reshape((1,img.shape[0],img.shape[1]))
        model.eval()
        with torch.no_grad():
            pred=model(img.to(device))
        _,predicted = torch.max(pred.data, 1)

        predicted="Cataract" if predicted==1 else "Normal"
        return predicted,imgo


# if __name__=="__main__":
    
#     m=predict_img("./models/cataract/data/ODIR-5K/ODIR-5K/Training Images/2111_left.jpg")
#     print(m.predict_cataract())







