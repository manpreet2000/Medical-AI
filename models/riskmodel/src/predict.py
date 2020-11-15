import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


loaded_model = pickle.load(open('models/riskmodel/weight/model.pkl', 'rb'))

class predict:
    def __init__(self,arr):

        """arr: [['Age', 'Diastolic BP', 'Poverty index', 'Race', 'Red blood cells',
       'Sedimentation rate', 'Serum Albumin', 'Serum Cholesterol',
       'Serum Iron', 'Serum Magnesium', 'Serum Protein', 'Sex', 'Systolic BP',
       'TIBC', 'TS', 'White blood cells', 'BMI', 'Pulse pressure']]"""

        self.arr=arr

    def predict_risk(self):
        ret= loaded_model.predict(self.arr)[0]
        return ret
    
if __name__=="__main__":
    p=predict([[35.0,92.0,126.0,2.0,77.7,12.0,5.0,165.0,135.0,1.37,7.6,2.0,142.0, 	323.0, 	41.8, 	5.8, 	31.109434, 	50.0]])
    print(p.predict_risk())

