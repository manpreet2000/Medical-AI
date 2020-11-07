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
        #print(image_file)
        if image_file:
            # shutil.rmtree(upload_folder)
            # os.makedirs(upload_folder)
            # image_loc=os.path.join(upload_folder,image_file.filename)
            # image_file.save(image_loc)
            # image_name = os.path.basename(image_loc)
            # image_name = image_name.split('.')[0]
            # print(image_name)
            # print(image_loc)
            
            # # plt.imshow(cv2.imread(image_loc))
            # # plt.show()
            # print(upload_folder+"/"+image_name+"NEW.png")
            npimg = np.fromstring(image_file.read(),np.uint8)
            classifier=predict.predict_img(npimg)
            uri=classifier.predict_image()
            # plt.imshow(pred,cmap='gray')
            # plt.show()
            
            # cv2.imwrite(upload_folder+"/"+image_name+"New.png",pred)
            # print(pred.shape)
            return render_template('/btindex.html',image_loc=uri)
    return render_template('/btindex.html',image_loc=None)


# if __name__ == '__main__':
#     brainapp.run(debug=True,port=8000)
