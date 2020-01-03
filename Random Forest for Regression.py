#!/usr/bin/env python
# coding: utf-8

# In[75]:


# Using Random Forest for the Regression Problem


# In[76]:


import pandas as pd
import numpy as np


# In[77]:


dataset = pd.read_csv("/home/garv/Desktop/petrol_consumption.csv")


# In[78]:


dataset.head()


# In[79]:


x = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values


# In[80]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.8, random_state= 0)


# In[81]:


# Feature Scaling

# the Average_Income field has values in the range of thousands while Petrol_tax has
# values in range of tens. Therefore, it would be beneficial to scale our data.
# To do so, we will use Scikit-Learn's StandardScaler class.


# In[82]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[85]:


# Train the Algorithm
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 200, random_state=0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)


# In[86]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:


# here i want to make a graph between the X-axis contains the number of estimators
# while the Y-axis contains the value for root mean squared error.

