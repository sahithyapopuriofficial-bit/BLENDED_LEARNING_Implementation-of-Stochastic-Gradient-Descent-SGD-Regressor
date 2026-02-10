# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset, remove unnecessary columns, convert categorical data to dummy variables, and separate features (X) and target (y).
2. Standardize both features and target values, then split the data into training and testing sets.
3. Train an SGD Regressor model on the training data and generate predictions on the test data.
4.Evaluate using MSE, MAE, R², display coefficients, and plot actual vs predicted values.
## Program:
```
/*
Program to implement SGD Regressor for linear regression.
Developed by: POPURI SAHITHYA
RegisterNumber:  212225240106
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
data=pd.read_csv('CarPrice_Assignment.csv')
print(data.head())
print(data.info())
data=data.drop(['CarName','car_ID'],axis=1)
data=pd.get_dummies(data,drop_first=True)
X=data.drop('price',axis=1)
y=data['price']
scaler = StandardScaler()
X=scaler.fit_transform(X)
y=scaler.fit_transform(np.array(y).reshape(-1,1))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
sgd_model=SGDRegressor(max_iter=1000,tol=1e-3)
sgd_model.fit(X_train,y_train)
y_pred=sgd_model.predict(X_test)
print('Name: Popuri Sahithya')
print('Reg No:212225240106')
print(f"{'MSE':}:{mean_squared_error(y_test,y_pred):}")
print(f"{'MAE':}:{mean_absolute_error(y_test,y_pred):}")
print(f"{'R-squared':}:{r2_score(y_test,y_pred):}")
print("\nModel Coefficients:")
print("Coefficients:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)
plt.scatter(y_test,y_pred)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()
```

## Output:

<img width="801" height="629" alt="image" src="https://github.com/user-attachments/assets/681b6202-33df-4ebc-a05c-665123b4d53b" />

<img width="514" height="721" alt="image" src="https://github.com/user-attachments/assets/bd252705-4ae2-4619-a310-4b230c66c551" />

<img width="869" height="362" alt="image" src="https://github.com/user-attachments/assets/0330e3b2-5862-477f-ba74-e95497f9b0c0" />

<img width="774" height="575" alt="image" src="https://github.com/user-attachments/assets/9ecd173b-c130-40fc-90c3-68989ce97af8" />

## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
