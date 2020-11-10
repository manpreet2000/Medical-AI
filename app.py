# if you get this error :
# Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized. 
# uncomment this code 
########################################### 
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' 
###########################################

from flask import Flask,render_template
from models.cataract.catapp import catapp
from models.pneumonia.pneapp import pneapp
from models.braintumor.brainapp import brainapp
app=Flask(__name__,template_folder="./templates",static_folder="./static")
app.register_blueprint(brainapp,url_prefix="/models/braintumor")
app.register_blueprint(catapp,url_prefix="/models/cataract")
app.register_blueprint(pneapp,url_prefix="/models/pneumonia")

@app.route("/")
def index():
    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)
