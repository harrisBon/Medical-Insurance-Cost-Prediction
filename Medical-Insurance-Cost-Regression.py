#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("insurance.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.corr()


# In[7]:


df.groupby('sex').describe().transpose()


# In[8]:


df.groupby('region').describe().transpose()


# In[9]:


df.groupby('smoker').describe().transpose()


# In[10]:


# We see that the charges are significantly higher for people who smoke
sns.pairplot(df,hue='smoker')


# In[11]:


df.isnull().sum()


# In[12]:


# change our object types
sex_dummies = pd.get_dummies(df['sex'],drop_first=True)
df = pd.concat([df.drop('sex',axis=1),sex_dummies],axis=1)
df.columns


# In[13]:


region_dummies = pd.get_dummies(df['region'],drop_first=True)
df = pd.concat([df.drop('region',axis=1),region_dummies],axis=1)
df.columns


# In[14]:


smoker_dummies = pd.get_dummies(df['smoker'],drop_first=True)
df = pd.concat([df.drop('smoker',axis=1),smoker_dummies],axis=1)
df.columns


# In[15]:


df.corr()


# In[16]:


plt.figure(figsize=(12,7))
sns.heatmap(df.corr(),annot=True,cmap='viridis')


# In[17]:


df.head()


# In[18]:


# Again we see that there is a high correlation with people that smoke and charges
# We are going to scale and then train our model
y=df['charges']
X=df.drop('charges',axis=1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=77)


# In[21]:


X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[22]:


X_train.shape , X_test.shape


# In[23]:


#Linear Regression
from sklearn.linear_model import LinearRegression


# In[24]:


lm = LinearRegression()


# In[25]:


lm.fit(X_train,y_train)


# In[26]:


predictionslm = lm.predict( X_test)


# In[27]:


plt.scatter(y_test,predictionslm)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[28]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictionslm))
print('MSE:', metrics.mean_squared_error(y_test, predictionslm))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictionslm)))
print('EVS:',  metrics.explained_variance_score(y_test, predictionslm))


# In[29]:


sns.distplot((y_test-predictionslm),bins=50);


# In[30]:


# Decision Tree
from sklearn.tree import DecisionTreeRegressor


# In[31]:


dtree = DecisionTreeRegressor()


# In[32]:


dtree.fit(X_train,y_train)


# In[33]:


predictionstree = dtree.predict(X_test)


# In[34]:


plt.scatter(y_test,predictionstree)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[35]:


print('MAE:', metrics.mean_absolute_error(y_test, predictionstree))
print('MSE:', metrics.mean_squared_error(y_test, predictionstree))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictionstree)))
print('EVS:',  metrics.r2_score(y_test, predictionstree))


# In[36]:


sns.distplot((y_test-predictionstree),bins=50);


# In[37]:


# Random Forest
from sklearn.ensemble import RandomForestRegressor


# In[38]:


rfg = RandomForestRegressor(n_estimators=500)


# In[39]:


rfg.fit(X_train,y_train)


# In[40]:


predictionsrfg = rfg.predict(X_test)


# In[41]:


plt.scatter(y_test,predictionsrfg)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[42]:


print('MAE:', metrics.mean_absolute_error(y_test, predictionsrfg))
print('MSE:', metrics.mean_squared_error(y_test, predictionsrfg))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictionsrfg)))
print('EVS:',  metrics.r2_score(y_test, predictionsrfg))


# In[43]:


sns.distplot((y_test-predictionsrfg),bins=50);


# In[44]:


# Support Vector Regressor
from sklearn.svm import SVR


# In[45]:


svr_model = SVR()


# In[46]:


svr_model.fit(X_train,y_train)


# In[47]:


predictionsSVR = svr_model.predict(X_test)


# In[48]:


plt.scatter(y_test,predictionsSVR)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[49]:


print('MAE:', metrics.mean_absolute_error(y_test, predictionsSVR))
print('MSE:', metrics.mean_squared_error(y_test, predictionsSVR))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictionsSVR)))
print('EVS:',  metrics.r2_score(y_test, predictionsSVR))


# In[50]:


# We have negative R2 score which tells that SVR is a really wrong model


# In[51]:


#Gradient Booster Regression
from sklearn.ensemble import GradientBoostingRegressor


# In[52]:


gbr = GradientBoostingRegressor(n_estimators=600)


# In[53]:


gbr.fit(X_train,y_train)


# In[54]:


predictionsgbr=gbr.predict(X_test)


# In[55]:


plt.scatter(y_test,predictionsgbr)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[56]:


print('MAE:', metrics.mean_absolute_error(y_test, predictionsgbr))
print('MSE:', metrics.mean_squared_error(y_test, predictionsgbr))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictionsgbr)))
print('EVS:',  metrics.r2_score(y_test, predictionsgbr))


# In[57]:


sns.distplot((y_test-predictionsgbr),bins=50);


# In[58]:


from sklearn.linear_model import Lasso


# In[59]:


las=Lasso()


# In[60]:


las.fit(X_train,y_train
    )


# In[61]:


predictionslas=las.predict(X_test)


# In[62]:


plt.scatter(y_test,predictionslas)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[63]:


print('MAE:', metrics.mean_absolute_error(y_test, predictionslas))
print('MSE:', metrics.mean_squared_error(y_test, predictionslas))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictionslas)))
print('EVS:',  metrics.r2_score(y_test, predictionslas))


# In[64]:


# Similar results with the Linear Regression Model
sns.distplot((y_test-predictionsgbr),bins=50);


# In[65]:


# ANN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam


# In[66]:


model = Sequential()

model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')


# In[67]:


model.fit(x=X_train,y=y_train.values,
          validation_data=(X_test,y_test.values),
          batch_size=128,epochs=500)


# In[68]:


losses = pd.DataFrame(model.history.history)
losses.plot()


# In[69]:


predictions = model.predict(X_test)


# In[70]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('EVS:',  metrics.r2_score(y_test, predictions))


# In[71]:


# It seems that the best models are the RandomForestRegressor and the GradientBoosterRegressor

