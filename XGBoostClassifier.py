import os
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



def data_preprocess(df):
    y = df["cancer_type"]
    label_encoder = LabelEncoder();
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

df=pd.read_csv("data.csv")
X, y=data_preprocess(df)

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

model=xgb.XGBClassifier(n_estimators=200, max_depth=5, tree_method='gpu_hist',random_state=42)

init_mod=model.fit(X_train,y_train)

y_pred=init_mod.predict(X_test)

print("acc=",accuracy_score(y_test,y_pred))
