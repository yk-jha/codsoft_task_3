import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

file_path = "advertising.csv"
data = pd.read_csv(file_path)

print(data.head())

#summary statistics of Data 
print(data.describe())

#check for missing values in data
print(data.isnull().sum())

#data types of each column
print(data.dtypes)

# handling missing values (if any)
data = data.dropna()

X = data.drop('Sales', axis=1)
y = data['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train , y_train)

y_pred_linear = model.predict(X_test)

mse_linear = mean_squared_error(y_test, y_pred_linear)

r2_linear = r2_score(y_test , y_pred_linear)

print(f'Linear Regression - Mean Squared Error: {mse_linear}')
print(f'Linear Regression - R-squared: {r2_linear}')

rf_model = RandomForestRegressor(random_state=42)

rf_model.fit(X_train , y_train)

y_pred_rf = rf_model.predict(X_test)

mse_rf = mean_squared_error(y_test , y_pred_rf)

r2_rf = r2_score(y_test , y_pred_rf)

print(f'Random Forest - Mean Squared Error: {mse_rf}')
print(f'Random Forest - r2 Score: {r2_rf}')

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f'Best parameters: {best_params}')

rf_model_tuned = RandomForestRegressor(**best_params)
rf_model_tuned.fit(X_train, y_train)

y_pred_rf_tuned = rf_model_tuned.predict(X_test)

mse_rf_tuned = mean_squared_error(y_test, y_pred_rf_tuned)
r2_rf_tuned = r2_score(y_test, y_pred_rf_tuned)

print(f'Tuned Random Forest - Mean Squared Error: {mse_rf_tuned}')
print(f'Tuned Random Forest - R-squared: {r2_rf_tuned}')

new_data = pd.DataFrame({
    'TV': [150],
    'Radio': [25],
    'Newspaper': [20]
})

predicted_sales = rf_model_tuned.predict(new_data)
print(f'Predicted Sales: {predicted_sales[0]}')

'''Summary:
This project aimed to predict sales based on advertising budgets using machine learning models. 
Linear Regression and Random Forest models were trained and evaluated, 
with the Random Forest model showing better performance. 
Hyperparameter tuning further improved the model's accuracy.

Key Findings:

The Random Forest model performed better than the Linear Regression model.
The tuned Random Forest model had the lowest Mean Squared Error and the highest R-squared value.


'''
