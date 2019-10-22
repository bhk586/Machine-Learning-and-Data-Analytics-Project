#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from scipy import stats
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)




np.random.seed(1234) 
import os 
os.environ['PYTHONHASHSEED']='1234' 
import random as rn
rn.seed(1234)


# In[18]:


#Pull excell Data via Pandas
#pd_data = pd.read_csv('AB_NYC_2019.csv', delimiter=',')
pd_data.head(3)


# ## Print Correlation Table and Graphs

# In[4]:


pd_data.corr()


# In[6]:


sns.pairplot(pd_data[['neighbourhood_group', 'neighbourhood', 'latitude', 'longitude', 'room_type', 'price', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'availability_365']]);


# ## Make descriptive values into dummy variables

# In[10]:


neighbourhood_groupDummy = pd.get_dummies(pd_data['neighbourhood_group'])
room_typeDummy = pd.get_dummies(pd_data['room_type'])
neighbourhoodDummy = pd.get_dummies(pd_data['neighbourhood'])
print(room_typeDummy.shape,
      neighbourhood_groupDummy.shape,
      neighbourhoodDummy.shape)


# In[16]:


pd_data = pd.concat([pd_data,neighbourhood_groupDummy],axis=1)
pd_data = pd.concat([pd_data,neighbourhoodDummy],axis=1)
pd_data = pd.concat([pd_data,room_typeDummy],axis=1)

print(pd_data.shape)


# ## Remove Attributes that were made into Dummy Variables

# In[36]:


#pd_data = pd_data.drop(['id', 'name', 'host_id', 'host_name', 'neighbourhood_group','neighbourhood', 'room_type', 'last_review'], axis=1)
pd_data.head()


# ## Make Panda Data into Numpy array for model

# In[42]:


#Export data into numpy arrays
np_data = pd_data.values

#Check that Data was imported correctly
print(np_data[0,])


# In[ ]:


x1 = np_data


# In[ ]:


y_test = np.array(pd_data['price'])
norm_ytest = preprocessing.normalize([y_test])


x1_test = np.array(pd_data['latitude'])
#norm_x1test = preprocessing.normalize([x1_test])

x2_test = np.array(pd_data['longitude'])
norm_x2test = preprocessing.normalize([x2_test])



#plt.scatter(norm_x1test, norm_ytest)
#plt.scatter(norm_x2test, norm_ytest)


# In[29]:


#Pullout Lat & long for model test
x_fake_train = np_data[1:101, 4:]
y_fake_train = np_data[1:101, 9]

#print(x_fake_train.shape,y_fake_train.shape)

ones = np.ones([x_fake_train.shape[0], 1])   #Add additional column to x Data for intercept
x_fake_train = np.concatenate((ones,x_fake_train),axis=1)

#Normalize
y_fake_train /= np.max(y_fake_train)
x_fake_train[:,1] /= np.max(x_fake_train[:,1])
x_fake_train[:,2] /= np.max(x_fake_train[:,2])

#Scale Data

#print(x_fake_train.shape,y_fake_train.shape)
#x_fake_train


# In[20]:


plt.scatter(x_fake_train[:,0], y_fake_train)
plt.scatter(x_fake_train[:,1], y_fake_train)
plt.scatter(x_fake_train[:,2], y_fake_train)
plt.ylabel('Pricing')


# In[27]:


#Define gradient descent

def Gradient_Descent(x_train, y_train, alpha, iterations):
    n = len(y_train)
    m = np.random.randn(3)
    Cost = np.zeros(iterations)
    for i in range(iterations):
        y_pred = x_train.dot(m)
        RSS = np.sum(y_train - y_pred)
        m = m - (alpha / n) * (y_train - y_pred).dot(x_train)
        Cost[i] = Cost_Funct(y_train, y_pred)
        if (i % 10 == 0):
            print(Cost[i])
        
    return m, Cost

#Define Cost Function

def Cost_Funct(y_train, y_pred):
    n = len(y_train)
    cost = (2 / n) * np.sum(np.power(y_train - y_pred, 2))
    return cost


# In[28]:


#Make linear regressiong model
theta = np.random.randn(3)

iterations = 100
alpha = 0.05

beta, Cost = Gradient_Descent(x_fake_train, y_fake_train, alpha, iterations)

plt.plot(range(iterations), Cost)

