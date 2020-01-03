#!/usr/bin/env python
# coding: utf-8

# In[87]:


# Using Random Forest for the Classification Problem


# In[88]:


import pandas as pd
import numpy as np


# In[89]:


dataset = pd.read_csv("/home/garv/Desktop/bill_authentication.csv")


# In[90]:


dataset.head()


# In[91]:


x = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values


# In[92]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8, random_state= 0)


# In[93]:


# Feature Scaling

# the Average_Income field has values in the range of thousands while Petrol_tax has
# values in range of tens. Therefore, it would be beneficial to scale our data.
# To do so, we will use Scikit-Learn's StandardScaler class.


# In[94]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[95]:


# Train the Algorithm
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, random_state=0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)


# In[98]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


# In[99]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[97]:


# here i want to make a graph between the X-axis contains the number of estimators
# while the Y-axis contains the value for root mean squared error.

