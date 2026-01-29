import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

 #Loading features into a DataFrame
X_train=pd.read_csv("data/X_train.csv")
y_train=pd.read_csv("data/y_train.csv").values.ravel()
X_test=pd.read_csv("data/X_test.csv") 
y_test=pd.read_csv("data/y_test.csv").values.ravel()  


model=RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train,y_train)

#Make prediction
y_pred=model.predict(X_test)

#evaluate the accuracy
print("Accuracy:", accuracy_score(y_test,y_pred))
print("Classification Report:\n", classification_report(y_test,y_pred))

#Save the model
joblib.dump(model,"model/churn_model.pkl")
joblib.dump(X_train.columns.tolist(),"model/feature_columns.pkl")
print("Model and feature columns saved successfully.")