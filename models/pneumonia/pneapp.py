import os
import shutil
from flask import Flask,request,render_template,Blueprint
import torch
import torch.nn as nn
from flask_cors import CORS
<<<<<<< HEAD
import base64
import numpy as np
from io import BytesIO
=======

>>>>>>> baa7f281db791260b5815b6c8781a5749ce3543d
# custom package
#import src.predict as predict 
import models.pneumonia.src.predict as predict
#pneapp = Flask(__name__)
pneapp=Blueprint("pneapp",__name__,template_folder="templates",static_folder="static")
CORS(pneapp)

upload_folder="./models/pneumonia/static"

@pneapp.route("/", methods=["GET","POST"])
def index():
    if request.method=="POST":
        image_file=request.files["file"]
        if image_file:
            npimg = np.fromstring(image_file.read(),np.uint8)
            classifier=predict.predict_img(npimg)
            result,img=classifier.predict_pneumonia()
            byteIO = BytesIO()
            img.save(byteIO, format="JPEG")
            img_base64 = base64.b64encode(byteIO.getvalue()).decode('ascii')
            mime = "image/jpeg"
            uri = "data:%s;base64,%s"%(mime, img_base64)
            
            return render_template('/pneindex.html',resultt=result,image_loc=uri)
    return render_template('/pneindex.html',resultt=None,image_loc=None)

# if __name__ == '__main__':
#     pneapp.run(debug=True,port=3000)