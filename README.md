# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
1.Import necessary libraries (e.g., NumPy, Pandas, Scikit-learn)
2.Load your dataset into a Pandas DataFrame
### Step2
1.Handle missing values (e.g., imputation, removal)
2.Scale/normalize features (e.g., StandardScaler)
### Step3
1.Split data into training (~70-80%) and testing sets (~20-30%)
2.Create a Multivariate Linear Regression model (e.g., Scikit-learn's LinearRegression)
### Step4
1.Fit the model to the training data
2.Evaluate the model using metrics like MSE, R-squared, and MAE
### Step5
1.Refine the model by tuning hyperparameters or feature engineering
2.Deploy the final model in your desired application or platform
## Program:
```
import pandas as pd
from sklearn import linear_model
df = pd.read_csv("C:\\Users\\admin\\Downloads\\car.csv")
X = df[['Weight', 'Volume']]
y = df['CO2']
regr = linear_model.LinearRegression()
regr.fit(X, y)
print('Coefficients:', regr.coef_)
print('Intercept:',regr.intercept_)
predictedCO2 = regr.predict([[3300, 1300]])
print('Predicted CO2 for the corresponding weight and volume',predictedCO2)
```
## Output:
![image](https://github.com/user-attachments/assets/b1210017-2e1e-4dc7-99d3-1ad491c1aec9)

### Insert your output

<br>

## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
