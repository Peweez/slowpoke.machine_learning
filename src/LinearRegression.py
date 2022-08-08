import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

# read .csv into a DataFrame
house_data = pd.read_csv('../resources/house_prices.csv')
size = house_data['sqft_living']
price = house_data['price']

# machine learning handle arrays not data-frames
x = np.array(size).reshape(-1, 1)
y = np.array(price).reshape(-1, 1)

# I use Linear Regression and fit() is the model training
model = LinearRegression()
model.fit(x, y)

# MSE and R value
regression_model_mse = mean_squared_error(x, y)
print('MSE: ', math.sqrt(regression_model_mse))
print('R squared value: ', model.score(x, y))

# I can get the b values after the model fit
# this is the b1
print(model.coef_[0])
# this is the b0 of the model
print(model.intercept_[0]) 

# visualize the dataset with  the fitted model
plt.scatter(x, y, color='green')
plt.plot(x, model.predict(x), color='black')
plt.title('Linear Regression')
plt.xlabel('Size')
plt.ylabel('Price')
plt.show()

# predicting the prices
print('Prediction by the model: ', model.predict([[2000]]))
