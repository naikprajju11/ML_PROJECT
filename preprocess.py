import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data/churn_prediction.csv')
print(data.shape)

# Convert 'TotalCharges' to numeric, forcing errors to NaN
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

#drop the rows with missing values
data.dropna(inplace=True)
print(f"Data shape after dropping missing values: {data.shape}")


duplicate=data.duplicated().sum()
print(f"Number of duplicate rows: {duplicate}")

if duplicate > 0:
    data.drop_duplicates()
    print(f"Data shape after dropping duplicates: {data.shape}")
    
    
if 'customerID' in data.columns:
    data.drop('customerID', axis=1, inplace=True)
    
    


numeric_cols=['tenure', 'MonthlyCharges', 'TotalCharges']

for col in numeric_cols:
    plt.figure(figsize=(8,4))
    sns.boxplot(x='Churn', y=col, data=data)
    plt.title(f'Boxplot of {col} by Churn')
    plt.show()
    

data['Churn']=data['Churn'].map({'Yes':1, 'No':0})

#encode categorical features
categorical_cols=data.select_dtypes(include=['object']).columns.tolist()
data=pd.get_dummies(data,columns=categorical_cols,drop_first=True)


#split into feature and targetX
X=data.drop('Churn',axis=1)
y=data['Churn']

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

