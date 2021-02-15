import pandas as pd

data = pd.read_csv('/Users/user/Downloads/CA_2013_onwards.csv')
data.head(5)

list(data)

# ## Deleting unusable/unnecessary columns

del data['id']
del data['state']
del data['stop_time']
del data['location_raw']
del data['county_fips']
del data['fine_grained_location']
del data['police_department']
del data['driver_age']
del data['driver_race_raw']
del data['violation_raw']
del data['search_type_raw']
del data['ethnicity']
del data['search_type']
list(data)
data.head(10)

## Making Arrest Dummie

dstop_outcome = pd.get_dummies(data.stop_outcome)
data=data.merge(dstop_outcome,left_index=True, right_index=True)
del data['CHP 215']
del data['CHP 281']
del data['CVSA Sticker']
del data['Motorist/Public Service']
del data['Traffic Collision']
del data['Turnover/Agency Assist']
del data['Verbal Warning']
data.head(10)
del data['stop_outcome']
del data['is_arrested']
data.head(10)


## Making Year Dummies and Season Dummies

from datetime import datetime
type('stop_date')

# Complete the call to convert the date column
data['stop_date'] =  pd.to_datetime(data['stop_date'],
                              format='%Y-%m-%d')
type('stop_date')
print(data['stop_date'][100000].day)

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

print(data.season.unique())

data['year']=data.stop_date.dt.year
dyear = pd.get_dummies(data.year)
data=data.merge(dyear, left_index=True, right_index=True)
dseason = pd.get_dummies(data.season)
data=data.merge(dseason, left_index=True, right_index=True)
list(data)

del data['stop_date']
del data['season']
del data['year']
print(data.info())

## Making Other Dummies

dcounty = pd.get_dummies(data.county_name)
data=data.merge(dcounty, left_index=True, right_index=True)
dgender = pd.get_dummies(data.driver_gender)
data=data.merge(dgender, left_index=True, right_index=True)
dage = pd.get_dummies(data.driver_age_raw)

data=data.merge(dage, left_index=True, right_index=True)
drace = pd.get_dummies(data.driver_race)

data=data.merge(drace, left_index=True, right_index=True)

dviolation = pd.get_dummies(data.violation)
data=data.merge(dviolation, left_index=True, right_index=True)
dconduct = pd.get_dummies(data.search_conducted)
data=data.merge(dconduct, left_index=True, right_index=True)
dfound = pd.get_dummies(data.contraband_found)
data=data.merge(dfound, left_index=True, right_index=True)

data.head(10)

del data['county_name']
del data['driver_gender']
del data['driver_age_raw']
del data['driver_race']
del data['violation']
del data['search_conducted']
del data['contraband_found']
data.head(10)
list(data)
del data['F']
del data['False_x']
del data['False_y']
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

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
from sklearn.tree import DecisionTreeClassifier

# Make a decision tree and train
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X, y)
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

# Random forest model creation
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)

# Training predictions
train_rf_predictions = rfc.predict(X_train)
train_rf_probs = rfc.predict_proba(X_train)[:, 1]

# Predictions to determine performance
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

# Splitting into training and test sample

from sklearn.model_selection import train_test_split
X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

# Random forest model creation
rfc = RandomForestClassifier()
rfc.fit(X1_train,y_train)

# Predictions to determine performance
rfc_predict = rfc.predict(X1_test)
rfc_pred_prob = rfc.predict_proba(X1_test)[:,1]

# Import scikit-learn metrics module for accuracy calculation
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

