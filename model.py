#!/usr/bin/env python
# coding: utf-8

# ## Predicting House Prices
# 
# > #### Task
# - The task is to predict house prices based on various features like the size of the house, number of bedrooms, and location, using a machine learning algorithm
# 
# 
# > #### Objectives
# 
#  - Understand how machine learning algorithms can be applied to predict continuous values (regression problem)
#  - Implement a Linear Regression model to predict house prices
# - Evaluate the model's performance using evaluation metrics such as Mean Absolute Error, Mean Squared Error, and R squared (R²)

# In[83]:


# importing neccessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score


# In[36]:


# loading the dataset
df = pd.read_csv("Data/train.csv")
df.head()


# In[38]:


print(f'we have {df.shape[0]} number of records and {df.shape[1]} number of features')


# In[40]:


#checking the dataset info
df.info()


# In[42]:


df.dtypes.value_counts()


# #### Data Preprocessing

# In[47]:


# Drop columns that are not useful for prediction
columns_to_drop = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']
df = df.drop (columns=columns_to_drop , axis=1)


# In[50]:


# Convert categorical variables into dummy variables
df = pd.get_dummies (df, drop_first =True)


# In[54]:


# Fill missing numerical values with the median of the respective columns
df = df.fillna(df.median())


# In[58]:


# Split the data into features (X) and target variable (y)
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']


# #### Split the Dataset into Training and Testing Sets

# In[64]:


# Use an 80 - 20 train test split
X_train, X_test , y_train , y_test = train_test_split (X, y, test_size =0.2, random_state =42)


# #### Train the Linear Regression Model

# In[68]:


# Initialize the Linear Regression model
model = LinearRegression()


# In[70]:


# Fit the model to the training data
model.fit(X_train , y_train)


# #### Making predictions with the linear Regression model

# In[73]:


# Predict on the test set
y_pred = model.predict(X_test)


# #### Evaluating the linear Regression model

# In[104]:


# Calculate evaluation metrics
mae = mean_absolute_error(y_test , y_pred)
mse= mean_squared_error(y_test , y_pred)
r2 = r2_score(y_test , y_pred)


# In[106]:


# Print the evaluation results
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R squared (R²): {r2}')


# #### VISUALIZE THE RESULTS

# In[81]:


# Plot actual vs predicted house prices
plt.figure(figsize =(8, 6))
plt.scatter(y_test , y_pred , alpha=0.6, color ='blue')
plt.plot([min(y_test), max(y_test)], [min(y_pred), max(y_pred)], color ='red', linewidth=2)
plt.title('Actual vs Predicted House Prices - Linear Regression')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()


# #### Using another regression model: Random Forest Regressor

# ##### Train the Random Forest Regression Model

# In[90]:


#Initialize the Random Forest Regressor
model =RandomForestRegressor(random_state =42)


# In[110]:


# Fit the model to the training data
model.fit(X_train , y_train)


# In[111]:


#### Making predictions with the Random Forest Regressor model
# Predict on the test set
y_pred = model.predict(X_test)


# In[112]:


#### Evaluating the Random Forest Regressor model
mae = mean_absolute_error(y_test , y_pred)
mse= mean_squared_error(y_test , y_pred)
r2 = r2_score(y_test , y_pred)

# Print the evaluation results
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R squared (R²): {r2}')


# In[ ]:

plt.figure(figsize =(8, 6))
plt.scatter(y_test , y_pred , alpha=0.6, color ='blue')
plt.plot([min(y_test), max(y_test)], [min(y_pred), max(y_pred)], color ='red', linewidth=2)
plt.title('Actual vs Predicted House Prices - Random Forest Regressor')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()


# === Export trained Random Forest model ===
import joblib
joblib.dump(model, r'model/random_forest_model.joblib')
print('Model saved to: model/random_forest_model.joblib')