import deep_wide
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


df=pd.read_csv('data.csv')
X, y=deep_wide.data_preprocess(df)

scaler=StandardScaler()

X=scaler.fit_transform(X)

summary = {
    'count': np.count_nonzero(~np.isnan(X), axis=0),
    'mean': np.nanmean(X, axis=0),
    'std': np.nanstd(X, axis=0),
    'min': np.nanmin(X, axis=0),
    '25%': np.nanpercentile(X, 25, axis=0),
    '50%': np.nanpercentile(X, 50, axis=0),
    '75%': np.nanpercentile(X, 75, axis=0),
    'max': np.nanmax(X, axis=0)
}

summary_df = pd.DataFrame(summary)

print(summary_df)

"""

X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.2)
dtrain=xgb.DMatrix(X_train, label=y_train)
dtest=xgb.DMatrix(X_test, label=y_test)
params={
	'objective':'multi:softmax',
	'num_class':4,
	'tree_method':'gpu_hist',
	'device':'cuda'
}
num_boost_round=110
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
"""
hyper_params={
	'booster':['gbtree'],
	'device':'cuda',
	'eta':[0.01,0.1,0.3,0.5,0.8,0.9],
	'gamma':[0.001,0.05,0.1,0.2,0.5],
	'max_depth':[3,6,9,12,15,21],
	'n_estimators':[100,200,300],
	'subsample':[0.3,0.5,0.8],
	'sampling_method':['uniform','gradient_based'],
	'lambda':[0.1,1,5,7,10,15],
	'tree_method':['gpu_hist','approx'],
	'updater':['grow_colmaker','grow_gpu_hist','grow_gpu_approx','prune'],
	'grow_policy':['depthwise','lossguide']

}

grid_search=GridSearchCV(estimator=model, param_grid=hyper_params, cv=5)

grid_search.fit(X_train,y_train)

best_model=grid_search.best_estimator

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the best accuracy and confusion matrix
print("Best Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
