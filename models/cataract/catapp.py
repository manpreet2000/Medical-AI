import os
import shutil
from flask import Flask,request,render_template
import torch
import torch.nn as nn
from flask_cors import CORS

# custom package
import src.predict as predict 


catapp=Flask(__name__)
CORS(catapp)

upload_folder="./models/cataract/static"

@catapp.route("/",methods=["GET","POST"])
def index():
    if request.method=="POST":
        image_file=request.files["file"]
        if image_file:
            shutil.rmtree(upload_folder)
            os.makedirs(upload_folder)
            image_loc=os.path.join(upload_folder,image_file.filename)
            image_file.save(image_loc)
            classifier=predict.predict_img(image_loc)
            result=classifier.predict_cataract()
            print(result)
            #print(image_loc)
            return render_template('index.html',resultt=result,image_loc=image_file.filename)
    return render_template('index.html',resultt=None,image_loc=None)

if __name__ == '__main__':
    catapp.run(debug=True,port=5000)
