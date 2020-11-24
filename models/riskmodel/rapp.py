from flask import Flask,render_template,Blueprint,request
import models.riskmodel.src.predict as predict
#import src.predict as predict
#rapp=Flask(__name__)
rapp=Blueprint("rapp",__name__,template_folder="templates",static_folder="static")
@rapp.route('/',methods=['POST','GET'])
def index():
    if request.method=='POST':
        age=float(request.form['age'])
        bp=float(request.form['Diastolic BP'])
        pi=float(request.form['Poverty index'])
        race=float(request.form['Race'])
        rbc=float(request.form['rbc'])
        sr=float(request.form['sr'])
        sa=float(request.form['sa'])
        sc=float(request.form['sc'])
        si=float(request.form['si'])
        sm=float(request.form['sm'])
        sp=float(request.form['sp'])
        sex=float(request.form['sex'])
        sbp=float(request.form['sbp'])
        tibc=float(request.form['TIBC'])
        ts=float(request.form['ts'])
        wbc=float(request.form['wbc'])
        bmi=float(request.form['bmi'])
        pp=float(request.form['pp'])
        inp=[]
        inp.append([age,bp,pi,race,rbc,sr,sa,sc,si,sm,sp,sex,sbp,tibc,ts,wbc,bmi,pp])
        model=predict.predict(inp)
        result=model.predict_risk()
        print(result)
        return render_template("/rkindex.html",resultt=result)
    return render_template("/rkindex.html",resultt=None)

# if __name__=="__main__":
#     rapp.run(debug=True)