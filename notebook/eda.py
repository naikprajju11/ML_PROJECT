import pandas as pd


data = pd.read_csv('data/churn_prediction.csv')
print(data.head())


print("Dataset info:")
print(data.info(), "\n")

#check for missing values   
print("Missing values in each column:")
print(data.isnull().sum(), "\n")


print("Statistical summary of numerical features:")
print(data.describe())


#target variable distribution
if "Churn" in data.columns:
    print("\nChurn distribution:")
    print(data['Churn'].value_counts())
else:
    print("\n'Churn' column not found in the dataset.")
    

categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
print("\n Unique values in categorical columns:")
for col in categorical_cols:
    print(f"{col}: {data[col].unique()}")
    
    
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
print('\n skewness of numerical features:')
print(data[numeric_cols].skew())
    