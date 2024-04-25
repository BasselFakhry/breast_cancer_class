import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

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

# Load data and preprocess
df = pd.read_csv('data.csv')
data_dmatrix,y = data_preprocess(df)

# Splitting data into train and test
X_train, X_test, y_train, y_test = train_test_split(data_dmatrix,y, test_size=0.2, random_state=42)

dtrain=xgb.DMatrix(data=X_train, label=y_train)
dtest=xgb.DMatrix(data=X_test, label=y_test)

# XGBoost parameters
params = {
    'objective': 'multi:softmax',
    'num_class': 4,
    'tree_method': 'hist',
    'eval_metric': 'mlogloss', 
    'device': 'cuda'
}

# Manual hyperparameter tuning setup
hyper_params = {
    'eta': [0.01, 0.1, 0.3, 0.5, 0.8, 0.9],
    'gamma': [0.001, 0.05, 0.1, 0.2, 0.5],
    'max_depth': [3, 6, 9, 12, 15, 21],
    'subsample': [0.3, 0.5, 0.8],
    'lambda': [0.1, 1, 5, 7, 10, 15]
}

# Function to perform custom hyperparameter tuning
def custom_hyperparam_tuning(params, hyper_params, data_dmatrix):
    best_params = None
    best_score = float('inf')

    for eta in hyper_params['eta']:
        for gamma in hyper_params['gamma']:
            for max_depth in hyper_params['max_depth']:
                for subsample in hyper_params['subsample']:
                    for lambda_ in hyper_params['lambda']:
                        params['eta'] = eta
                        params['gamma'] = gamma
                        params['max_depth'] = max_depth
                        params['subsample'] = subsample
                        params['lambda'] = lambda_
                        cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=5,
                                            num_boost_round=50, early_stopping_rounds=10, metrics='mlogloss', as_pandas=True)
                        mean_mlogloss = cv_results['test-mlogloss-mean'].min()
                        if mean_mlogloss < best_score:
                            best_score = mean_mlogloss
                            best_params = params.copy()

    return best_params, best_score

# Perform hyperparameter tuning
best_params, best_score = custom_hyperparam_tuning(params, hyper_params, dtrain)
print("Best Parameters:", best_params)
print("Best Score:", best_score)