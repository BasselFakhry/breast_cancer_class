import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def data_preprocess(df):
    # Isolate target and encode it
    y = df["cancer_type"]
    label_encoder_y = LabelEncoder()
    y = label_encoder_y.fit_transform(y)
    y = pd.Series(y)
    df = df.drop('cancer_type', axis=1)

    # Manually mapping and encoding categorical data
    categorical_mappings = {
        'cellularity': {'Low': 1, 'Moderate': 2, 'High': 3},
        'her2_status_measured_by_snp6': {'LOSS': 0, 'NEUTRAL': 1, 'GAIN': 3, 'UNDEF': 1},
        'primary_tumor_laterality': {'Right': 1, 'Left': -1},
    }
    
    for col, mapping in categorical_mappings.items():
        df[col] = df[col].str.strip().map(mapping).fillna(0)

    # Drop patient_id as it is irrelevant for modeling
    df = df.drop('patient_id', axis=1)

    # One-hot encoding other categorical columns
    categorical_columns = ['pam50_+_claudin-low_subtype', 'er_status', 'er_status_measured_by_ihc',
                           'her2_status', 'inferred_menopausal_state', 'pr_status', '3-gene_classifier_subtype',
                           'death_from_cancer']
    df = pd.get_dummies(df, columns=categorical_columns)

    # Handle missing values for numeric columns
    numeric_imputations = {
        'tumor_size': df['tumor_size'].mean(),
        'mutation_count': df['mutation_count'].mean(),
        'neoplasm_histologic_grade': 3,  # Mode or specific domain-related value
    }

    for col, value in numeric_imputations.items():
        df[col] = df[col].fillna(value)

    # Ensure all remaining categorical columns are encoded
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    return df, y

# Define data types for mixed type columns
data_types = {662: str, 664: str, 676: str, 677: str, 683: str, 685: str, 686: str, 687: str}
df = pd.read_csv('data.csv', dtype=data_types)

X, y = data_preprocess(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Model Definition
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y)), activation='softmax')  # Output layer nodes equal to number of classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
