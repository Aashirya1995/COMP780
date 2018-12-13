
# coding: utf-8

# The dataset used in this notebook could be found on this link: https://archive.ics.uci.edu/ml/datasets/Air+Quality

# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Atributes info:
# 
#     0 Date	(DD/MM/YYYY) 
#     1 Time	(HH.MM.SS) 
#     2 True hourly averaged concentration CO in mg/m^3 (reference analyzer) 
#     3 PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted)	
#     4 True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer) 
#     5 True hourly averaged Benzene concentration in microg/m^3 (reference analyzer) 
#     6 PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)	
#     7 True hourly averaged NOx concentration in ppb (reference analyzer) 
#     8 PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted) 
#     9 True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)	
#     10 PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)	
#     11 PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted) 
#     12 Temperature in Â°C	
#     13 Relative Humidity (%) 
#     14 AH Absolute Humidity 

# In[3]:


air_data = pd.read_csv('AirQualityUCI.csv', encoding='utf-8')


# In[4]:


air_data.head()


# In[5]:


air_data.shape


# In[6]:


type(air_data)


# In[7]:


air_data.keys()


# In[8]:


air_data.isnull().sum()


# In[9]:


corr_matrix = air_data.corr()
corr_matrix["CO(GT)"].sort_values(ascending=False)


# In[10]:


df1 = air_data[['CO(GT)','NO2(GT)']]
df1.head()


# In[11]:


import seaborn as sns
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
sns.set(style='whitegrid', context='notebook')
features_plot = ['C6H6(GT)', 'RH', 'AH', 'PT08.S1(CO)']

data_to_plot = air_data[features_plot]
data_to_plot = scalar.fit_transform(data_to_plot)
data_to_plot = pd.DataFrame(data_to_plot)

sns.pairplot(data_to_plot, size=2.0);
plt.tight_layout()
plt.show()


# In[15]:


sns.countplot(air_data['CO(GT)'])


# In[16]:


sns.countplot(air_data['NO2(GT)'])


# ## Preprocessing data

# In[11]:


air_data.dropna(axis=0, how='all')


# ## Features vs Labels

# In[12]:


features = air_data


# In[13]:


features = features.drop('Date', axis=1)
features = features.drop('Time', axis=1)
features = features.drop('C6H6(GT)', axis=1)
features = features.drop('PT08.S4(NO2)', axis=1)


# In[14]:


labels = air_data['C6H6(GT)'].values


# In[15]:


features = features.values


# ## Train and test portions

# In[16]:


from sklearn.cross_validation import train_test_split


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)


# In[18]:


print("X_trian shape {}".format(X_train.shape))
print("y_train shape {}".format(y_train.shape))
print("X_test shape  {}".format(X_test.shape))
print("y_test shape  {}".format(y_test.shape))


# ### Linear Regression

# In[19]:


from sklearn.linear_model import LinearRegression


# In[20]:


regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[21]:


print("Predicted values:", regressor.predict(X_test))


# In[22]:


print("R^2 score for liner regression: ", regressor.score(X_test, y_test))


# In[23]:


print('Coefficients: \n', regressor.coef_)


# In[24]:


print("Mean squared error: %.2f"
      % mean_squared_error(y_test, regressor.predict(X_test) ))


# ### Support Vector Regression

# In[25]:


from sklearn.cross_validation import KFold
from sklearn.svm import SVR


# In[26]:


support_regressor = SVR(kernel='rbf', C=1000)
support_regressor.fit(X_train, y_train)


# In[27]:


print("Coefficient of determination R^2 <-- on train set: {}".format(support_regressor.score(X_train, y_train)))


# In[28]:


print("Coefficient of determination R^2 <-- on test set: {}".format(support_regressor.score(X_test, y_test)))


# ### Decision tree regression

# In[29]:


from sklearn.tree import DecisionTreeRegressor


# In[30]:


dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)


# In[31]:


print("Coefficient of determination R^2 on train set: {}".format(dtr.score(X_train, y_train)))


# In[32]:


print("Coefficient of determination R^2 on test set: {}".format(dtr.score(X_test, y_test)))

