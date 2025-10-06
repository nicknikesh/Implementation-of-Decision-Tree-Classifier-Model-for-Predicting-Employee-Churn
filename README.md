# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import and Load Data: Import necessary libraries (pandas, sklearn) and load the employee churn dataset.

2.Preprocess Data: Handle missing values, encode categorical variables, and separate features (X) and target (y).

3.Split Data: Divide the dataset into training and testing sets using train_test_split().

4.Train Model: Create and train a DecisionTreeClassifier on the training data.

5.Predict and Evaluate: Use the model to predict churn on test data and evaluate performance using accuracy and confusion matrix.

## Program:

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by: NIKESH KUMAR C

RegisterNumber: 212223040132
```
import pandas as pd 
import numpy as np
df=pd.read_csv("Employee.csv")
print(df.head())
df.info()
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["salary"]=le.fit_transform(df["salary"])
df.head()
df["left"].value_counts()
x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=df["left"]
y.head()
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(X_train,Y_train)
y_pred=dt.predict(X_test)
print("Name:NIKESH KUMAR")
print("RegNo: 212223040132")
print(y_pred)
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(Y_test,y_pred)
cm=confusion_matrix(Y_test,y_pred)
cr=classification_report(Y_test,y_pred)
print("Accuracy:",accuracy)
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(cr)
dt.predict(pd.DataFrame([[0.6,0.9,8,292,6,0,1,2]],columns=["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]))
```

## Output:
![WhatsApp Image 2025-10-06 at 11 13 05_a12c8d13](https://github.com/user-attachments/assets/0d4f95e5-4442-475b-9b4c-7d7266945629)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
