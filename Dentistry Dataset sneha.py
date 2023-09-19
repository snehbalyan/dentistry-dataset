#sneha Balyan

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Dentistry Dataset.csv")
df.head()
df.shape
df.dtypes
df.info()
df.isnull().sum()
df.duplicated().sum()
df.duplicated().value_counts()

plt.figure(figsize = (9,6))
matrix = df.corr()
sns.heatmap(matrix,cmap='coolwarm',fmt = '.0%',annot = True,linewidths=0.5)
matrix

drop_column_list = ["Sl No","Sample ID"]

df.drop(drop_column_list,inplace = True,axis = 1)

df.info()

print(df['Age'].unique())

import warnings
warnings.filterwarnings('ignore')
sns.countplot(x='Gender',data=df)

df.describe().T

df.head(10)
df.shape

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Gender']

fig = plt.figure(figsize = (9,6))
sns.boxplot(data = df, linewidth = 1)
plt.xticks(rotation = ('vertical'))
plt.show()

def outliers (df,ft):
    Q1 = df[ft].quantile(0.25)
    Q3 = df[ft].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1-1.5*IQR
    upper_bound = Q3+1.5*IQR
    
    ls = df.index[(df[ft] < lower_bound) | (df[ft] > upper_bound)]
    return ls

def remove (df,ls):
    ls = sorted(set(ls))
    df = df.drop(ls)    
    return df

index_list=[]
for feature in ['Age','Gender','inter canine distance intraoral',
                'intercanine distance casts','right canine width intraoral',
                'right canine width casts','left canine width intraoral',
               'left canine width casts','right canine index intra oral',
               'right canine index casts','left canine index intraoral',
               'left canine index casts']:
    index_list.extend(outliers(df,feature))
    
df1 = remove(df,index_list)

df1.shape

fig = plt.figure(figsize = (9,6))
sns.boxplot(data = df1, linewidth = 1)
plt.xticks(rotation = ('vertical'))
plt.show()

df1.info()

# feature selection
x = df1.drop('Gender', axis = 1)
y = df1['Gender']

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

select_kbest_rank_feature = SelectKBest(score_func= chi2, k = 5)
kbest_feature = select_kbest_rank_feature.fit(x, y)

df_score = pd.DataFrame(kbest_feature.scores_,columns = ['Score'])
dfcolumns = pd.DataFrame(x.columns)

kbest_rank_feature_concat = pd.concat([dfcolumns,df_score], axis = 1)
kbest_rank_feature_concat.columns = ['features','k_score']
kbest_rank_feature_concat

print(kbest_rank_feature_concat.nlargest(15,'k_score'))

# drop the least correlated features
# check the correlation between independant variables
# split the data in to train and test
# perform logistic regression

#drop columns through creating list
K_Best_drop_features=['right canine index intra oral','left canine index casts',
                      'left canine index intraoral','right canine index casts']

df1.drop(K_Best_drop_features,inplace=True,axis=1)
df1.info()

plt.figure(figsize=(7,4))
matrix1 = df1.corr()
sns.heatmap(df1.corr(),annot = True,linewidth = 1,cmap = 'coolwarm',fmt = '.0%')
matrix1

#===================================================================

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.70)

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features = correlation(x_train, 0.80)
print(corr_features)
len(set(corr_features))

x_train = x_train.drop(corr_features, axis =1)
x_test = x_test.drop(corr_features, axis =1)

'''
#Making models to avoid MultiCollinearity issues among Independent Variables

x1 = df1[['inter canine distance intraoral','intercanine distance casts',
          'right canine width intraoral','right canine index intra oral']]

x2 = df1[['inter canine distance intraoral','intercanine distance casts',
          'right canine width casts','right canine index intra oral']]

x3 = df1[['inter canine distance intraoral','intercanine distance casts',
          'left canine width intraoral','right canine index intra oral']]

x4 = df1[['inter canine distance intraoral','intercanine distance casts',
          'left canine width casts','right canine index intra oral']]
'''

# 1. logistic regression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.75)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train,y_train)

y_train_pred = logreg.predict(x_train)
y_test_pred = logreg.predict(x_test)

from sklearn.metrics import accuracy_score
print("Training Accuracy Score:",accuracy_score(y_train,y_train_pred).round(3))
print("Test Accuracy Score:",accuracy_score(y_test,y_test_pred).round(3))

'''
#Model X2
from sklearn.model_selection import train_test_split
x2_train,x2_test,y_train,y_test = train_test_split(x2,y,train_size = 0.75)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x2_train,y_train)

y_train_pred = logreg.predict(x2_train)
y_test_pred = logreg.predict(x2_test)

from sklearn.metrics import accuracy_score
print("Training Accuracy Score:",accuracy_score(y_train,y_train_pred).round(3))
print("Test Accuracy Score:",accuracy_score(y_test,y_test_pred).round(3))

#Model X3
from sklearn.model_selection import train_test_split
x3_train,x3_test,y_train,y_test = train_test_split(x3,y,train_size = 0.75)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x3_train,y_train)

y_train_pred = logreg.predict(x3_train)
y_test_pred = logreg.predict(x3_test)

from sklearn.metrics import accuracy_score
print("Training Accuracy Score:",accuracy_score(y_train,y_train_pred).round(3))
print("Test Accuracy Score:",accuracy_score(y_test,y_test_pred).round(3))

#Model X4
from sklearn.model_selection import train_test_split
x4_train,x4_test,y_train,y_test = train_test_split(x4,y,train_size = 0.75)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x4_train,y_train)

y_train_pred = logreg.predict(x4_train)
y_test_pred = logreg.predict(x4_test)

from sklearn.metrics import accuracy_score
print("Training Accuracy Score:",accuracy_score(y_train,y_train_pred).round(3))
print("Test Accuracy Score:",accuracy_score(y_test,y_test_pred).round(3))
'''
#=====================================================================
# 2. confusion matrix

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_train,y_train_pred)
cm2 = confusion_matrix(y_test,y_test_pred)
print(cm1)
print(cm2)

from sklearn.metrics import classification_report
print(classification_report(y_train,y_train_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_test_pred))

# 2. confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm1 = confusion_matrix(y_train,y_train_pred)
print("Accuracy Score:",accuracy_score(y_train,y_train_pred).round(2))
TN=cm1[0,0]
FP=cm1[0,1]
TNR=TN/(TN+FP)
print("Specificity Score:",TNR.round(2))
from sklearn.metrics import recall_score,f1_score,precision_score
print("Sensitivity score:",recall_score(y_train,y_train_pred).round(2))
print("Precision Score:",precision_score(y_train,y_train_pred).round(2))
print("F1 Score:",f1_score(y_train,y_train_pred).round(2))
cm1

from sklearn.metrics import confusion_matrix,accuracy_score
cm2 = confusion_matrix(y_test,y_test_pred)
print("Accuracy Score:",accuracy_score(y_test,y_test_pred).round(2))
TN=cm2[0,0]
FP=cm2[0,1]
TNR=TN/(TN+FP)
print("Specificity Score:",TNR.round(2))
from sklearn.metrics import recall_score,f1_score,precision_score
print("Sensitivity score:",recall_score(y_test,y_test_pred).round(2))
print("Precision Score:",precision_score(y_test,y_test_pred).round(2))
print("F1 Score:",f1_score(y_test,y_test_pred).round(2))
cm2


#=======================================================================
# 3. k fold method
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits=10)
logreg = LogisticRegression()
results = cross_val_score(logreg,x,y,cv = kfold,scoring='accuracy')
np.mean(results).round(3)

'''
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits=10)
logreg = LogisticRegression()
results = cross_val_score(logreg,x2,y,cv = kfold,scoring='accuracy')
np.mean(results).round(3)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits=10)
logreg = LogisticRegression()
results = cross_val_score(logreg,x3,y,cv = kfold,scoring='accuracy')
np.mean(results).round(3)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits=10)
logreg = LogisticRegression()
results = cross_val_score(logreg,x4,y,cv = kfold,scoring='accuracy')
np.mean(results).round(3)
'''
#===========================================================================
# 4. loocv method
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
loocv = LeaveOneOut()
logreg = LogisticRegression()
results = cross_val_score(logreg,x,y,cv = loocv,scoring='accuracy')
np.mean(results).round(3)

'''
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
loocv = LeaveOneOut()
logreg = LogisticRegression()
results = cross_val_score(logreg,x2,y,cv = loocv,scoring='accuracy')
np.mean(results).round(3)

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
loocv = LeaveOneOut()
logreg = LogisticRegression()
results = cross_val_score(logreg,x3,y,cv = loocv,scoring='accuracy')
np.mean(results).round(3)

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
loocv = LeaveOneOut()
logreg = LogisticRegression()
results = cross_val_score(logreg,x4,y,cv = loocv,scoring='accuracy')
np.mean(results).round(3)
'''

# 5. Support Vecor Machine (SVM)

from sklearn.model_selection._split import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,
                                                 random_state=10)

from sklearn.svm import SVC
svc_linear = SVC(kernel='linear')
svc_linear.fit(x_train, y_train)
y_pred_train = svc_linear.predict(x_train)
y_pred_test  = svc_linear.predict(x_test)

from sklearn import metrics
print("Training Accuracy :",metrics.accuracy_score(y_train, y_pred_train).round(3))
print("Test Accuracy:",metrics.accuracy_score(y_test, y_pred_test).round(3))

# 6. Decission Tree with entropy/Gini index:

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy", max_depth = None)
dt.fit(x_train,y_train)

y_pred_train = dt.predict(x_train)
y_pred_test = dt.predict(x_test)

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_train,y_pred_train)
print("Training Accuracy score" ,ac1.round(3))
ac2 = accuracy_score(y_test,y_pred_test)
print("Test Accuracy score" ,ac2.round(3))

# 7. Random Forest:

from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(max_depth = 5,
                             n_estimators = 500,
                             max_samples = 0.6,max_features = 0.5,
                             bootstrap = True,random_state = 100)

RFC.fit(x_train,y_train)

y_pred_train = RFC.predict(x_train)
y_pred_test = RFC.predict(x_test)

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_train,y_pred_train)
print("Training Accuracy score" ,ac1.round(3))
ac2 = accuracy_score(y_test,y_pred_test)
print("Test Accuracy score" ,ac2.round(3))

# 8. Bagging method:

from sklearn.ensemble import BaggingClassifier
dt = DecisionTreeClassifier(max_depth = 5)
bag = BaggingClassifier(base_estimator = dt,
                        n_estimators = 100,
                        max_samples = 0.6,max_features = 0.5,
                        bootstrap = False)

bag.fit(x_train,y_train)

y_pred_train = bag.predict(x_train)
y_pred_test = bag.predict(x_test)

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_train,y_pred_train)
print("Training Accuracy score" ,ac1.round(3))
ac2 = accuracy_score(y_test,y_pred_test)
print("Test Accuracy score" ,ac2.round(3))

# 9. AdaBoost Algorithm

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
dtr = DecisionTreeRegressor()
ADB = AdaBoostRegressor(learning_rate = 2,n_estimators = 200,
                        base_estimator = dtr,random_state = 100)

ADB.fit(x_train,y_train)

y_pred_train = ADB.predict(x_train)
y_pred_test = ADB.predict(x_test)


from sklearn.metrics import mean_squared_error
mse1 = mean_squared_error(y_train,y_pred_train)
mse2 = mean_squared_error(y_test,y_pred_test)

RMSE1 = np.sqrt(mse1)
print("ADB-Training error",RMSE1.round(3))

RMSE2 = np.sqrt(mse2)
print("ADB-Test error",RMSE2.round(3))
