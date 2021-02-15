
# coding: utf-8

# In[ ]:


import pandas as pd


# In[2]:


data = pd.read_csv('/Users/user/Downloads/CA_2013_onwards.csv')
data.head(5)


# In[3]:


list(data)


# ## Deleting unusable/unnecessary columns

# In[4]:


del data['id']


# In[5]:


del data['state']


# In[6]:


del data['stop_time']


# In[7]:


del data['location_raw']


# In[8]:


del data['county_fips']


# In[9]:


del data['fine_grained_location']


# In[10]:


del data['police_department']


# In[11]:


del data['driver_age']


# In[12]:


del data['driver_race_raw']


# In[13]:


del data['violation_raw']


# In[14]:


del data['search_type_raw']


# In[15]:


del data['ethnicity']


# In[16]:


del data['search_type']


# In[17]:


list(data)


# In[18]:


data.head(10)


# ## Making Arrest Dummie

# In[19]:


dstop_outcome = pd.get_dummies(data.stop_outcome)


# In[20]:


data=data.merge(dstop_outcome,left_index=True, right_index=True)


# In[21]:


del data['CHP 215']


# In[22]:


del data['CHP 281']


# In[23]:


del data['CVSA Sticker']


# In[24]:


del data['Motorist/Public Service']


# In[25]:


del data['Traffic Collision']


# In[26]:


del data['Turnover/Agency Assist']


# In[27]:


del data['Verbal Warning']


# In[28]:


data.head(10)


# In[29]:


del data['stop_outcome']


# In[30]:


del data['is_arrested']


# In[31]:


data.head(10)


# ## Making Year Dummies and Season Dummies

# In[32]:


from datetime import datetime


# In[33]:


type('stop_date')


# In[34]:


# Complete the call to convert the date column
data['stop_date'] =  pd.to_datetime(data['stop_date'],
                              format='%Y-%m-%d')


# In[35]:


type('stop_date')


# In[36]:


print(data['stop_date'][100000].day)


# In[37]:


import datetime as dt

for i in range(2013, 2018):
    spring_begin, spring_end = dt.datetime(i, 3, 21), dt.datetime(i, 6, 21)
    summer_begin, summer_end = dt.datetime(i, 6, 22), dt.datetime(i, 9, 23)
    fall_begin, fall_end = dt.datetime(i, 9, 24), dt.datetime(i, 12, 21)
    winter_begin1, winter_end1 = dt.datetime(i, 1, 1), dt.datetime(i, 3, 20)
    winter_begin2, winter_end2 = dt.datetime(i, 12, 22), dt.datetime(i, 12, 31)
    
    data.loc[(data.stop_date>= spring_begin) & (data.stop_date<= spring_end), 'season' ] = 'spring'
    data.loc[(data.stop_date>= summer_begin) & (data.stop_date<= summer_end),'season' ] = 'summer'
    data.loc[(data.stop_date>= fall_begin) & (data.stop_date<= fall_end),'season' ] = 'fall'
    data.loc[(data.stop_date>= winter_begin1) & (data.stop_date<= winter_end1) , 'season' ] = 'winter'
    data.loc[(data.stop_date>= winter_begin2) & (data.stop_date<= winter_end2) , 'season' ] = 'winter'


# In[38]:


print(data.season.unique())


# In[39]:


data['year']=data.stop_date.dt.year


# In[40]:


dyear = pd.get_dummies(data.year)


# In[41]:


data=data.merge(dyear, left_index=True, right_index=True)


# In[42]:


dseason = pd.get_dummies(data.season)


# In[43]:


data=data.merge(dseason, left_index=True, right_index=True)


# In[44]:


list(data)


# In[45]:


del data['stop_date']


# In[46]:


del data['season']


# In[47]:


del data['year']


# In[48]:


print(data.info())


# ## Making Other Dummies

# In[49]:


dcounty = pd.get_dummies(data.county_name)


# In[50]:


data=data.merge(dcounty, left_index=True, right_index=True)


# In[51]:


dgender = pd.get_dummies(data.driver_gender)


# In[52]:


data=data.merge(dgender, left_index=True, right_index=True)


# In[53]:


dage = pd.get_dummies(data.driver_age_raw)


# In[54]:


data=data.merge(dage, left_index=True, right_index=True)


# In[55]:


drace = pd.get_dummies(data.driver_race)


# In[56]:


data=data.merge(drace, left_index=True, right_index=True)


# In[57]:


dviolation = pd.get_dummies(data.violation)


# In[58]:


data=data.merge(dviolation, left_index=True, right_index=True)


# In[59]:

dconduct = pd.get_dummies(data.search_conducted)

# In[60]:

data=data.merge(dconduct, left_index=True, right_index=True)

# In[61]:

dfound = pd.get_dummies(data.contraband_found)


# In[62]:


data=data.merge(dfound, left_index=True, right_index=True)


# In[63]:


data.head(10)


# In[64]:


del data['county_name']


# In[65]:


del data['driver_gender']


# In[66]:


del data['driver_age_raw']


# In[67]:


del data['driver_race']


# In[68]:


del data['violation']


# In[69]:


del data['search_conducted']


# In[70]:


del data['contraband_found']


# In[71]:


data.head(10)


# In[72]:


list(data)


# In[73]:


del data['F']


# In[74]:


del data['False_x']


# In[75]:


del data['False_y']


# In[76]:


print(data.info())

data.to_csv('finalcaStop.csv')

## Random Forest Classification

import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
data = pd.read_csv('/Users/user/Downloads/finalcastop.csv')

data.describe()

del data['Unnamed: 0']
pd.set_option('display.max_columns', 83)
data.describe()

list(data)

pd.set_option('display.max_columns', 83)

data.describe()

X.head(5)

X.iloc[:,78]

X = data.drop('Arrest', axis=1)
y = data['Arrest']

# In[22]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# In[7]:
from sklearn.tree import DecisionTreeClassifier
# Make a decision tree and train
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X, y)

# In[23]:

from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)

# Training predictions
train_rf_predictions = rfc.predict(X_train)
train_rf_probs = rfc.predict_proba(X_train)[:, 1]

# predictions to determine performance
rfc_predict = rfc.predict(X_test)
rfc_pred_prob = rfc.predict_proba(X_test)[:,1]

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, rfc_predict))

import pandas as pd
feature_imp = pd.Series(rfc.feature_importances_,).sort_values(ascending=False)
feature_imp

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score
roc_value = roc_auc_score(y_test, rfc_pred_prob)
rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc')

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", roc_value.mean())
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())

X.head(8)

X1 = X.iloc[:, [78 ,74, 75, 62, 79, 5, 7, 1, 2, 0, 77, 6, 4, 76]]

# splitting into training and test sample
from sklearn.model_selection import train_test_split

X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)

from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(X1_train,y_train)

# predictions to determine performance
rfc_predict = rfc.predict(X1_test)
rfc_pred_prob = rfc.predict_proba(X1_test)[:,1]

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, rfc_predict))

import pandas as pd
feature_imp = pd.Series(rfc.feature_importances_,).sort_values(ascending=False)
feature_imp

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score
roc_value = roc_auc_score(y_test, rfc_pred_prob)
rfc_cv_score = cross_val_score(rfc, X1, y, cv=10, scoring='roc_auc')

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", roc_value.mean())
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, rfc_predict))

import pandas as pd
feature_imp = pd.Series(rfc.feature_importances_,).sort_values(ascending=False)
feature_imp

