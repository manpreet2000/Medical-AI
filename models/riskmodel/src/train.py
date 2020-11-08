import pandas as pd
import numpy as np
import config
import utils
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lifelines
from sklearn.model_selection import GridSearchCV
import pickle


x,y=utils.load(10)
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=10)

imputer = IterativeImputer(random_state=0, sample_posterior=False, max_iter=1, min_value=0)
imputer.fit(X_train)
X_train_imputed = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
X_val_imputed = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)

parameters = {'criterion':['gini','entropy'], 'max_depth':[1,2,3,10],
                           
                           'n_estimators':[1,10,100,120,150,170,200,500],
                           }

rf = RandomForestClassifier(random_state=10)
clf = GridSearchCV(rf, parameters)
clf.fit(X_train_imputed, y_train)

y_train_rf_preds = clf.predict_proba(X_train_imputed)[:, 1]
print(f"Train C-Index: {utils.cindex(y_train.values, y_train_rf_preds)}")

y_val_rf_preds = clf.predict_proba(X_val_imputed)[:, 1]
print(f"Val C-Index: {utils.cindex(y_val.values, y_val_rf_preds)}")

pickle.dump(clf, open(config.model_dir+'model.pkl', 'wb'))

