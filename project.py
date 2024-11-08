import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py

data = pd.read_csv('/home/troy/Downloads/uk_car_dataset/bmw.csv')

nnpDf = data.sort_values("price", ascending=False).iloc[108:]
nnpDf.drop('transmission', axis=1, inplace=True)
nnpDf.drop('model', axis=1, inplace=True)
nnpDf.drop('fuelType', axis=1, inplace=True)
df = nnpDf
y = df['price'].values
x = df.drop('price', axis=1).values
print(x)
print("\n", y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(x_train, y_train)
y_pred_lin_reg = linear_regression.predict(x_test)


from sklearn.tree import DecisionTreeRegressor
decision_tree_reg = DecisionTreeRegressor(random_state=0)
decision_tree_reg.fit(x_train,y_train)
y_pred_lin_reg = decision_tree_reg.predict(x_test)

from sklearn.ensemble import RandomForestRegressor
random_forest_reg = RandomForestRegressor(n_estimators = 50, random_state=0)
random_forest_reg.fit(x_train, y_train)
y_pred_rf_reg = random_forest_reg.predict(x_test)

print(f'y_test: {y_test}\nprediction: {y_pred_lin_reg}')
print(random_forest_reg.predict([[2000, 90000, 290, 25.5, 3.8]]))