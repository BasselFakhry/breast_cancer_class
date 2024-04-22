import xgboost as xgb
import deeb_wide
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df=pd.read_csv('data.csv')

X, y=deep_wide.data_preprocess(df)

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

model=xgb.XGBCLassifier(n_estimators=200, max_depth=5, tree_method='gpu_hist')

init_mod=model.fit(X_train,y_train)

y_pred=init_model.predict(X_test)

print("acc=",accuracy_score(y_test,y_pred_))
