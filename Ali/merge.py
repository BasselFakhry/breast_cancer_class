#import pandas as pd

    # Load the datasets from CSV files
    # Replace 'path/to/dataset1.csv' and 'path/to/dataset2.csv' with your actual file paths
#df1 = pd.read_csv('data.csv')
#df2 = pd.read_csv('clean.csv')

    # Merging the datasets
    # This keeps all rows from df2 and adds 'cancer_type' from df1 where 'age_at_diagnosis' matches
#result = pd.merge(df2, df1[['age_at_diagnosis', 'cancer_type']], on='age_at_diagnosis', how='left')

    # Save the result to a new CSV file
    # Replace 'path/to/merged_dataset.csv' with your desired output file path
#result.to_csv('dataset.csv', index=False)

    # Optional: Print the first few rows to check the result
#print(result.head())

