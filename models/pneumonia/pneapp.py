import os
import shutil
from flask import Flask,request,render_template,Blueprint
import torch
import torch.nn as nn
from flask_cors import CORS

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
            shutil.rmtree(upload_folder)
            os.makedirs(upload_folder)
            image_loc=os.path.join(upload_folder,image_file.filename)
            image_file.save(image_loc)
            classifier=predict.predict_img(image_loc)
            result=classifier.predict_pneumonia()
            print(result)
            #print(image_loc)
            return render_template('/pneindex.html',resultt=result,image_loc=image_file.filename)
    return render_template('/pneindex.html',resultt=None,image_loc=None)

# if __name__ == '__main__':
#     pneapp.run(debug=True,port=3000)