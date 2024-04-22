import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def data_preprocess(df):
    y = df["cancer_type"]
    label_encoder = LabelEncoder()
    y  = label_encoder.fit_transform(y)
    y = pd.Series(y)
    df = df.drop('cancer_type', axis = 1)

    # label encoding for cellularity 40 nan values transformed to 0
    mapping = {
        'Low': 1,
        'Moderate': 2,
        'High': 3,
    }
    df['cellularity'] = df['cellularity'].str.strip()
    df["cellularity"] = df["cellularity"].map(mapping)
    df["cellularity"] = df["cellularity"].fillna(0)


    # dropping patient_id (irrelevant info)
    df = df.drop('patient_id', axis=1)

    #label encoding pam50_+_claudin-low_subtype
    df['pam50_+_claudin-low_subtype'] =label_encoder.fit_transform( df['pam50_+_claudin-low_subtype'])

    df['er_status'] =label_encoder.fit_transform( df['er_status'])

    df['er_status_measured_by_ihc'] = label_encoder.fit_transform(df['er_status_measured_by_ihc'])

    df['her2_status'] = label_encoder.fit_transform(df['her2_status'])

    her2_mapping={
    'LOSS' : 0,
    'NEUTRAL' : 1,
    'GAIN' : 3,
    'UNDEF' : 1
    }

    df['her2_status_measured_by_snp6'] = df['her2_status_measured_by_snp6'].str.strip()
    df['her2_status_measured_by_snp6'] = df['her2_status_measured_by_snp6'].map(her2_mapping)

    df['inferred_menopausal_state'] = label_encoder.fit_transform(df['inferred_menopausal_state'])

    map_laterality = {
    'Right':1,
    'Left':-1,
    }
    df['primary_tumor_laterality'] = df['primary_tumor_laterality'].str.strip()
    df['primary_tumor_laterality'] = df['primary_tumor_laterality'].map(map_laterality)
    df['primary_tumor_laterality'] = df['primary_tumor_laterality'].fillna(0)

    df['pr_status'] = label_encoder.fit_transform(df['pr_status'])

    df = pd.get_dummies(df, columns=['3-gene_classifier_subtype'])

    df = pd.get_dummies(df, columns=['death_from_cancer'])

    tumor_mean = df['tumor_size'].mean()
    df["tumor_size"] = df["tumor_size"].fillna(tumor_mean)

    mutation_mean = df['mutation_count'].mean()
    df['mutation_count'] = df['mutation_count'].fillna(mutation_mean)

    df['neoplasm_histologic_grade'] = df['neoplasm_histologic_grade'].fillna(3)

    majority_value = df['tumor_stage'].mode()[0]
    df['tumor_stage'].fillna(majority_value, inplace=True)
    df['tumor_stage']=label_encoder.fit_transform(df['tumor_stage'])

    label_encoders = {}

    for column in df.columns:
        if df[column].dtype == 'object':
            # Create a label encoder for each categorical column
            le = LabelEncoder()

            # Fit the label encoder and transform the data
            df[column] = le.fit_transform(df[column].astype(str))

            # Store the label encoder in a dictionary in case you need to reverse the encoding or use it later
            label_encoders[column] = le
    
    last_seven = df.iloc[:, -7:]
    part_before = df.iloc[:, :2]  # Columns up to the 19th (0-based index, so it includes columns 0-18)
    part_after = df.iloc[:, 2:]
    df = pd.concat([part_before, last_seven, part_after], axis=1)
    df = df.iloc[:, :-7]


    
    return df,y


df=pd.read_csv('data.csv')
X, y=data_preprocess(df)



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

# predictions=model.predict(dtest)

# cm = confusion_matrix(y_test, predictions)

# tn =cm[0,0]
# fp=cm[0,1]
# fn=cm[1,0]
# tp=cm[1,1]

# labels = ['True pos','False pos','true neg','false neg']

# print("Confusion matrix: ")
# print(f"{labels[0]}: {tn}, {labels[1]}: {fp}")
# print(f"{labels[2]}: {fn}, {labels[3]}: {tp}")


# Print accuracy score
#print("Accuracy Score:", accuracy_score(y_test,predictions))


hyper_params={
    'objective':['multi:softmax','multi:softprob'],
    'eval_metric':['rmse','rmsle','logloss'],
	'eta':[0.01,0.1,0.3,0.5,0.8,0.9],
	'gamma':[0.001,0.05,0.1,0.2,0.5],
	'max_depth':[3,6,9,12,15,21],
	'n_estimators':[100,200,300],
    'num_parallel_tree':[1,3,5,10,20,30],
	'subsample':[0.3,0.5,0.8],
	'sampling_method':['uniform','gradient_based'],
	'lambda':[0.1, 0.5 , 1, 5, 7,10],
    'alpha':[0.01,0.05,0.1,0.3,0.5,0.8,1],
	'tree_method':['hist','approx'],
	'grow_policy':['depthwise','lossguide'],
    'updater':['grow_histmaker','grow_quantile_histmaker','grow_hist']

}

xgb_clf = xgb.XGBClassifier(objective='multi:softmax', num_class=4,device='cuda',process_type='update')

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator=xgb_clf, param_distributions=hyper_params, n_iter=100, scoring='accuracy', cv=5)

# Perform the random search
random_search.fit(X_train,y_train)

# Get the best parameters and best score
best_params = random_search.best_params_
best_score = random_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)
