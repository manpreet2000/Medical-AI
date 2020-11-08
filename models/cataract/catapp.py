import os
import shutil
from flask import Flask,Blueprint,request,render_template,url_for
import torch
import torch.nn as nn
from flask_cors import CORS
import base64
import numpy as np
from io import BytesIO
# custom package
#import src.predict as predict 
import models.cataract.src.predict as predict

# catapp=Flask(__name__)
catapp=Blueprint("catapp",__name__,template_folder="templates",static_folder="static")
#CORS(catapp)


@catapp.route("/",methods=["GET","POST"])
def index():
    if request.method=="POST":
        image_file=request.files["file"]
        if image_file:
            npimg = np.fromstring(image_file.read(),np.uint8)
            classifier=predict.predict_img(npimg)
            result,img=classifier.predict_cataract()
            byteIO = BytesIO()
            img.save(byteIO, format="JPEG")
            img_base64 = base64.b64encode(byteIO.getvalue()).decode('ascii')
            mime = "image/jpeg"
            uri = "data:%s;base64,%s"%(mime, img_base64)

            return render_template('/catindex.html',resultt=result,image_loc=uri)
    return render_template('/catindex.html',resultt=None,image_loc=None)

# if __name__ == '__main__':
#     catapp.run(debug=True,port=5000)
