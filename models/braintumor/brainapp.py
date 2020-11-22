import os
from flask import Flask, request, render_template,url_for,Blueprint
from flask_cors import CORS, cross_origin
import shutil
import models.braintumor.src.predict as predict 
import base64
import numpy as np
from io import BytesIO
#brainapp = Flask(__name__)
brainapp=Blueprint("brainapp",__name__,template_folder="templates",static_folder="static")
#CORS(brainapp)

upload_folder="./models/braintumor/static"


@brainapp.route("/", methods=["GET","POST"])
def index():
    if request.method=="POST":
        image_file=request.files["file"]
        
        if image_file:
            
            npimg = np.fromstring(image_file.read(),np.uint8)
            classifier=predict.predict_img(npimg)
            uri=classifier.predict_image()
            
            return render_template('/btindex.html',image_loc=uri)
    return render_template('/btindex.html',image_loc=None)


# if __name__ == '__main__':
#     brainapp.run(debug=True,port=8000)
