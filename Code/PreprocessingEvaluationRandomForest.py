#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:01:22 2024

@author: mali
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

df = pd.read_csv('/Users/mali/Downloads/winequality-red.csv')


#%% check NaN values and duplicates
print(df.isnull().sum())
print(df.duplicated())

df = df.drop_duplicates()

#%% Detecting Outliers
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest()

iso_forest.fit(df)

outliers = iso_forest.predict(df)

plt.scatter(df.iloc[:,0],df.iloc[:,1],c=outliers,cmap='coolwarm')
plt.title("IsolationForest Outliers")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
#%% removing Outliers
df = df[outliers==1]


#%% Seperating input and output

x = df.iloc[:,:11].to_numpy()
y = df.iloc[:,11].to_numpy()


#%% Data Normalization

# we have many normalization techniques.I'm going to show some of them

# 1) Min-Max Scaling

x_norm1 = (x - np.min(x))/(np.max(x)-np.min(x))

# 1) Z-Score Normalization (Standardization)
# we can implement this method with 2 way
# First with scikit learn

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_norm2 = scaler.fit_transform(x)

# Second Without Scikit Learn so With Python

x_norm_manuel = np.zeros_like(x)
for i in range(x.shape[1]):
    mean = np.mean(x[:,i])
    std = np.std(x[:,i])
    x_norm_manuel[:,i] = (x[:,i] - mean) / std
    

#%% Check is target set imbalanced?
class_count = np.bincount(y)

if class_count.max() / class_count.sum() > 0.7:
    print('Data set is imblanced')
    
else:
    print('Data set is not imbalnced')
    
#%% split train and set data

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_norm2,y,test_size=0.15,random_state=7)

#%% fit our model

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()

dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100,random_state=7)

rf.fit(x_test,y_test)
y_pred2 = rf.predict(x_test)

rf.score(x_test,y_test)

#%% Evalution 
# R-Aquared
from sklearn.metrics import r2_score

r2 = r2_score(y_test,y_pred2)

# Adjusted R-Squared

def adjusted(x,r2):
    n = x.shape[0]
    p = x.shape[1]
    
    adj = 1 - ((1-r2)*(n-1))/(n-p-1)
    
    return adj 

print(adjusted(x_test, r2))






























