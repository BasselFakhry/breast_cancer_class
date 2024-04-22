import deep_wide
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np


df=pd.read_csv('data.csv')
X, y=deep_wide.data_preprocess(df)


X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.2)
dtrain=xgb.DMatrix(X_train, label=y_train)
dtest=xgb.DMatrix(X_test, label=y_test)
params={
	'objective':'multi:softmax',
	'num_class':4,
	'tree_method':'gpu_hist',
	'device':'cuda'
}
num_boost_round=100
model=xgb.train(params, dtrain, num_boost_round=num_boost_round)

predictions=model.predict(dtest)

cm = confusion_matrix(y_test, predictions)

tn =cm[0,0]
fp=cm[0,1]
fn=cm[1,0]
tp=cm[1,1]

labels = ['True pos','False pos','true neg','false neg']

print("Confusion matrix: ")
print(f"{labels[0]}: {tn}, {labels[1]}: {fp}")
print(f"{labels[2]}: {fn}, {labels[3]}: {tp}")


# Print accuracy score
print("Accuracy Score:", accuracy_score(y_test,predictions))  
