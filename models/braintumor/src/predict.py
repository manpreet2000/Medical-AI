import torch
import torch.nn as nn
import os
import albumentations as A
import numpy as np
from collections import OrderedDict
import cv2
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensor
import io
import base64
from PIL import Image

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        #print(x.shape)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        #print(dec1.shape)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )



class predict_img:
    def __init__(self,image_code):
        self.image_code=image_code
        self.test_transforms = A.Compose([
                A.Resize(width = 256, height = 256, p=1.0),
                A.Normalize(p=1.0),
                ToTensor(),
            ])

    def predict_image(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model=UNet()
        print("\n Model found! Loading \n")
        model.load_state_dict(torch.load("./models/braintumor/weights/model.h5", map_location=lambda storage, loc: storage))
        model=model.to(device)
        img=cv2.imdecode(self.image_code,cv2.IMREAD_COLOR)
        imgo = Image.fromarray(img.astype("uint8"))
        img=np.array(imgo)
        img_p=self.test_transforms(image=img)['image']
        img_p=img_p.reshape((1,3,256,256))
        pred_o=model(img_p.to(device).float())
        pred_o=pred_o.detach()
        pred_o=pred_o.reshape((256,256))


        plt.subplot(1,2,1)
        
        plt.imshow(img)
        plt.title('Original Image')
        plt.subplot(1,2,2)
        plt.imshow(pred_o.numpy(),cmap='gray')
        plt.title('Tumor Segmentation')
        buf = io.BytesIO()
        plt.savefig(buf, format="jpg", dpi=180)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        
        img_base64 = base64.b64encode(buf.getvalue()).decode('ascii')
        mime = "image/jpeg"
        uri = "data:%s;base64,%s"%(mime, img_base64)
        return uri
       


# if __name__=="__main__":
#     c=predict_img("./brain tumor/static/TCGA_HT_8111_19980330_9.tif","","")
#     c.predict_image()

    
#     #print(pred.min(),pred.max(),pred.mean())
#     #print(pred.shape)
