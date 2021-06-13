import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,make_scorer
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV

df = pd.read_csv('weatherAUS.csv')

df.head()

df.shape

df.info()

df.describe()

# Finding cols with lots of NA values
df.isna().sum()

"""Here we are seeing that, there are four features which contains around (45-50)% of missing value"""

sns.set_style('darkgrid')
sns.set_palette('plasma_r')
plt.figure(figsize=[15,6])
ax = sns.countplot(x = 'RainTomorrow',edgecolor=(0,.78,1),linewidth=3,data = df)
ax.set_title( "Occurence of rain (before data cleaning)",size = 30 )
plt.show()
print("RainTomorrow value for Yes: ", len(df.loc[df['RainTomorrow']=="Yes"]),'and for No:',len(df.loc[df['RainTomorrow']=="No"]))

# There is class imbalance in non cleaned df

df['RainTomorrow'].isna().sum()

numerical = [var for var in df.columns if df[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)

df_num = df[numerical]

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} numerical variables\n'.format(len(categorical)))

print('The numerical variables are :', categorical)
df_cat = df[categorical]

""" Plot correlation matrix with target variable RainTomorrow """
df_na_cols= df[['MinTemp','MaxTemp','Rainfall','Evaporation', 'Sunshine', 'Cloud9am','Cloud3pm','Pressure9am','Pressure3pm','Humidity9am','Humidity3pm','Temp9am','Temp3pm','WindGustSpeed','WindSpeed9am',
   'WindSpeed3pm','RainTomorrow']]
df_na_cols.head()

df_na_cols['RainTomorrow'] = df_na_cols['RainTomorrow'].fillna('No')
df_na_cols['RainTomorrow'].isna().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_na_cols['RainTomorrow'] = le.fit_transform(df_na_cols['RainTomorrow'])

#Used correlation plot to analyse the impact of these features on our target variable "RainTomorrow"
sns.heatmap(df_na_cols.corr(), annot=True, fmt='.2f')
sns.heatmap(df_na_cols.corr(), annot=True, fmt='.2f')
# Cloud9am, Cloud3pm, Humidity9am , Humidity3pm are closely releated to our target variable and there is a good relationship between them also.
# Fetures like Pressure9am, Pressure3pm, Evaporation and Sunshine etc are badly related with target and between themselves also.

#Droping columns whose correlation is not significant with the target variable
df = df.drop(['Evaporation','Sunshine','WindGustDir','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Pressure9am','Pressure3pm','Temp9am','Temp3pm','MaxTemp'], axis = 1)

#Updating the Cloud9am and Cloud3pm cols with mean value as they show +ve relation- can be altered with using filling na rows with mean values
df['Cloud9am'].fillna(df.Cloud9am.mean())
df['Cloud3pm'].fillna(df.Cloud3pm.mean())
#dropping rows that contain na values- tried using filling methods such as avg of ffill and bfill but giving less f1 score
df.dropna(inplace = True)
np.round(df.isnull().sum())

# Data is cleaned and ready in df
df.info()

#Checking for class balance after cleaning
sns.set_style('darkgrid')
sns.set_palette('plasma_r')
plt.figure(figsize=[15,6])
ax = sns.countplot(x = 'RainTomorrow',edgecolor=(0,.78,1),linewidth=3,data = df)
ax.set_title( "Occurence of rain (after data cleaning)",size = 30 )
plt.show()
print("RainTomorrow value for Yes: ", len(df.loc[df['RainTomorrow']=="Yes"]),'and for No:',len(df.loc[df['RainTomorrow']=="No"]))
#It shows that there is class imbalance - F1 score should be a good evaluator

""" Let's have look at our variable

Variable Types

1. Numerical variables:
   'MinTemp', 'MaxTemp', 'Rainfall','WindGustSpeed','WindSpeed9am',
   'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
   'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm',

2. Categorical variables:
   'Date', 'Location','WindGustDir','WindDir9am','WindDir3pm','RainToday', 'RainTomorrow'



# Feature Selection

Rain tomorrow is strongly related to rain today as shown
"""

# This shows that if it rains today then the possibility is high for raining the next day based on data stats

plt.figure(figsize = (8,4))
sns.countplot(x = 'RainToday', hue = 'RainTomorrow', data = df)

plt.figure(figsize = (40,40))
sns.heatmap(df.corr(), cmap = 'RdBu', annot = True, linewidths=1, linecolor='black')

""" Modeling the data """

#Encoding Yes and no to 1 and 0 and one hot encoding of numerical features and dropping date col
le = LabelEncoder()
df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
df['RainTomorrow'] = le.fit_transform(df['RainTomorrow'])
df = pd.get_dummies(df,
                    #columns=['Location','WindGustDir','WindDir9am','WindDir3pm','RainToday'],
                    columns=['Location','RainToday'],
                    drop_first=True)
df.drop(['Date'],inplace=True,axis=1)
df.info()

df.head()

# initialize results arrays
model = []
precision = []
recall = []
F1score = []
Accuracy = []
AUCROC = []

# drop target variable
X = df.drop(['RainTomorrow'], axis = 1)
y = df['RainTomorrow']

# Splitting up the data to test and train set
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.30, random_state=42)

# data scaling for better performance
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(X_train)
# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


"""LogisticRegression:"""

Log = GridSearchCV(estimator=LogisticRegression(max_iter=1000),
					param_grid={'class_weight': [{0:1, 1:v} for v in np.linspace(1,20,30)]},
					scoring={'precision':make_scorer(precision_score), 'recall': make_scorer(recall_score)},
					refit='precision',
					return_train_score=True,
					cv=10,
					n_jobs=-1)

Log.fit(X_train, y_train)

y_prob=Log.predict_proba(X_test)[:,1]
y_pred=Log.predict(X_test)
#ROC curve
fpr, tpr, thresholds  = roc_curve(y_test, y_prob)

plt.rcParams['font.size'] = 12
plt.plot(fpr, tpr, color = 'blue', marker = 'o', markerfacecolor = 'red', markersize = 1)
plt.plot([0,1], [0,1], 'k--' )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC Curve for Logistic Regression')
plt.xlabel('False Positive Rate(1-Specificity)')
plt.ylabel('True Positive Rate(Sensitivity)')
plt.grid(True)
plt.show()

print('='*60)
print("\nPerformance Metrics for Logistic Regression\n")
print('='*60)
print('Confusion Matrix')
print(confusion_matrix(y_test,y_pred),"\n")
print('-'*30)
print('Classification Report')
print(classification_report(y_test,y_pred),"\n")
print('-'*30)
print('AUC-ROC')
print(roc_auc_score(y_test, y_prob))
print('-'*30)
          
model.append('Logistic Regression')
precision.append(precision_score(y_test,y_pred))
recall.append(recall_score(y_test,y_pred))
F1score.append(f1_score(y_test,y_pred))
Accuracy.append(accuracy_score(y_test, y_pred))
AUCROC.append(roc_auc_score(y_test, y_prob))

"""**Naive Bayes Classifier:**

GaussianNB:
"""

from sklearn.naive_bayes import GaussianNB
Naive_bayes= GaussianNB()

Naive_bayes.fit(X_train, y_train)

y_prob=Naive_bayes.predict_proba(X_test)[:,1]
y_pred=Naive_bayes.predict(X_test)

fpr, tpr, thresholds  = roc_curve(y_test, y_prob)

plt.rcParams['font.size'] = 12
plt.plot(fpr, tpr, color = 'blue', marker = 'o', markerfacecolor = 'red', markersize = 1)
plt.plot([0,1], [0,1], 'k--' )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC Curve for GaussianNB')
plt.xlabel('False Positive Rate(1-Specificity)')
plt.ylabel('True Positive Rate(Sensitivity)')
plt.grid(True)
plt.show()

print('='*60)
print("\nPerformance Metrics for GaussianNB\n")
print('='*60)
print('Confusion Matrix')
print(confusion_matrix(y_test,y_pred),"\n")
print('-'*30)
print('Classification Report')
print(classification_report(y_test,y_pred),"\n")
print('-'*30)
print('AUC-ROC')
print(roc_auc_score(y_test, y_prob))
print('-'*30)
          
model.append('GaussianNB')
precision.append(precision_score(y_test,y_pred))
recall.append(recall_score(y_test,y_pred))
F1score.append(f1_score(y_test,y_pred))
Accuracy.append(accuracy_score(y_test, y_pred))
AUCROC.append(roc_auc_score(y_test, y_prob))


"""**Weighted Decision Tree**"""

from sklearn.tree import DecisionTreeClassifier
#DT=DecisionTreeClassifier() #'class_weight': [{0:1, 1:v} for v in np.linspace(1,20,30)]
DT = GridSearchCV(estimator=DecisionTreeClassifier(criterion='entropy'),
					param_grid={'max_depth' : [v for v in range(2,8)]},
					scoring={'precision':make_scorer(precision_score), 'recall': make_scorer(recall_score)},
					refit='precision',
					return_train_score=True,
					cv=10,
					n_jobs=-1)
DT.fit(X_train, y_train)
#print(DT.best_params_)
y_prob=DT.predict_proba(X_test)[:,1]
y_pred=DT.predict(X_test)

fpr, tpr, thresholds  = roc_curve(y_test, y_prob)

plt.rcParams['font.size'] = 12
plt.plot(fpr, tpr, color = 'blue', marker = 'o', markerfacecolor = 'red', markersize = 1)
plt.plot([0,1], [0,1], 'k--' )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC Curve for Decision Tree Classifier')
plt.xlabel('False Positive Rate(1-Specificity)')
plt.ylabel('True Positive Rate(Sensitivity)')
plt.grid(True)
plt.show()

print('='*60)
print("\nPerformance Metrics for DecisionTreeClassifier\n")
print('='*60)
print('Confusion Matrix')
print(confusion_matrix(y_test,y_pred),"\n")
print('-'*30)
print('Classification Report')
print(classification_report(y_test,y_pred),"\n")
print('-'*30)
print('AUC-ROC')
print(roc_auc_score(y_test, y_prob))
print('-'*30)

model.append('Decision Tree')
precision.append(precision_score(y_test,y_pred))
recall.append(recall_score(y_test,y_pred))
F1score.append(f1_score(y_test,y_pred))
Accuracy.append(accuracy_score(y_test, y_pred))
AUCROC.append(roc_auc_score(y_test, y_prob))


"""**Random Forest/ Bagging**"""

from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier()
clf = BaggingClassifier(n_estimators=10, random_state=0).fit(X_train, y_train)
y_prob=clf.predict_proba(X_test)[:,1]
y_pred=clf.predict(X_test)

fpr, tpr, thresholds  = roc_curve(y_test, y_prob)

plt.rcParams['font.size'] = 12
plt.plot(fpr, tpr, color = 'blue', marker = 'o', markerfacecolor = 'red', markersize = 1)
plt.plot([0,1], [0,1], 'k--' )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC Curve for BaggingClassifier')
plt.xlabel('False Positive Rate(1-Specificity)')
plt.ylabel('True Positive Rate(Sensitivity)')
plt.grid(True)
plt.show()

print('='*60)
print("\nPerformance Metrics for BaggingClassifier\n")
print('='*60)
print('Confusion Matrix')
print(confusion_matrix(y_test,y_pred),"\n")
print('Classification Report')
print('-'*30)
print(classification_report(y_test,y_pred),"\n")
print('AUC-ROC')
print('-'*30)
print(roc_auc_score(y_test, y_prob))
print('-'*30)

model.append('BaggingClassifier')
precision.append(precision_score(y_test,y_pred))
recall.append(recall_score(y_test,y_pred))
F1score.append(f1_score(y_test,y_pred))
Accuracy.append(accuracy_score(y_test, y_pred))
AUCROC.append(roc_auc_score(y_test, y_prob))

""" K-NN """

#Add dynamic k value and check
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=9)

# uncomment below multi-line comments to choose best K value via GridSearch and Cross-validation
"""neigh = GridSearchCV(estimator=KNeighborsClassifier(),
					param_grid={'n_neighbors':[v for v in range(1,10,2)]},
					scoring={'precision':make_scorer(precision_score), 'recall': make_scorer(recall_score)},
					refit='precision',
					return_train_score=True,
					cv=5,
					n_jobs=-1 """

neigh.fit(X_train, y_train)
#print(neigh.best_params_)

y_prob=neigh.predict_proba(X_test)[:,1]
y_pred=neigh.predict(X_test)

fpr, tpr, thresholds  = roc_curve(y_test, y_prob)

plt.rcParams['font.size'] = 12
plt.plot(fpr, tpr, color = 'blue', marker = 'o', markerfacecolor = 'red', markersize = 1)
plt.plot([0,1], [0,1], 'k--' )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC Curve for k-NN')
plt.xlabel('False Positive Rate(1-Specificity)')
plt.ylabel('True Positive Rate(Sensitivity)')
plt.grid(True)
plt.show()

print('='*60)
print("\nPerformance Metrics for k-NN\n")
print('='*60)
print('Confusion Matrix')
print('-'*30)
print(confusion_matrix(y_test,y_pred),"\n")
print('Classification Report')
print('-'*30)
print(classification_report(y_test,y_pred),"\n")
print('AUC-ROC')
print(roc_auc_score(y_test, y_prob))
print('-'*30)

model.append('k-NN')
precision.append(precision_score(y_test,y_pred))
recall.append(recall_score(y_test,y_pred))
F1score.append(f1_score(y_test,y_pred))
Accuracy.append(accuracy_score(y_test, y_pred))
AUCROC.append(roc_auc_score(y_test, y_prob))



"""  Linear SVM Classifier  """

from sklearn.svm import LinearSVC
svm = LinearSVC(C=1) #'class_weight': [{0:1, 1:v} for v in np.linspace(1,10,5)],

svm.fit(X_train, y_train)
#print(svm.best_params_)

#y_prob=svm.predict_proba(X_test)[:,1]
y_pred=svm.predict(X_test)

"""fpr, tpr, thresholds  = roc_curve(y_test, y_prob)

plt.rcParams['font.size'] = 12
plt.plot(fpr, tpr, color = 'blue', marker = 'o', markerfacecolor = 'red', markersize = 1)
plt.plot([0,1], [0,1], 'k--' )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC Curve for Linear SVM CLassifier')
plt.xlabel('False Positive Rate(1-Specificity)')
plt.ylabel('True Positive Rate(Sensitivity)')
plt.grid(True)
plt.show() """

print('='*60)
print("\nPerformance Metrics for Linear SVM\n")
print('='*60)
print('Confusion Matrix')
print(confusion_matrix(y_test,y_pred),"\n")
print('-'*30)
print('Classification Report')
print(classification_report(y_test,y_pred),"\n")
print('-'*30)
#print('AUC-ROC')
#print(roc_auc_score(y_test, y_prob))
#print('-'*30)

model.append('Linear SVM')
precision.append(precision_score(y_test,y_pred))
recall.append(recall_score(y_test,y_pred))
F1score.append(f1_score(y_test,y_pred))
Accuracy.append(accuracy_score(y_test, y_pred))
AUCROC.append(roc_auc_score(y_test, y_prob))

""" Stochastic GD - with Linear SVM """

from sklearn.linear_model import SGDClassifier
svm = SGDClassifier()
svm.fit(X_train, y_train)
#y_prob=svm.predict_proba(X_test)[:,1]
y_pred=svm.predict(X_test)

fpr, tpr, thresholds  = roc_curve(y_test, y_prob)

plt.rcParams['font.size'] = 12
plt.plot(fpr, tpr, color = 'blue', marker = 'o', markerfacecolor = 'red', markersize = 1)
plt.plot([0,1], [0,1], 'k--' )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC Curve for SGD CLassifier')
plt.xlabel('False Positive Rate(1-Specificity)')
plt.ylabel('True Positive Rate(Sensitivity)')
plt.grid(True)
plt.show()

print('='*60)
print("\nPerformance Metrics for Stochastic Gradient Decision\n")
print('='*60)
print('Confusion Matrix')
print(confusion_matrix(y_test,y_pred),"\n")
print('-'*30)
print('Classification Report')
print(classification_report(y_test,y_pred),"\n")
print('-'*30)
#print('AUC-ROC')
#print(roc_auc_score(y_test, y_prob))
#print('-'*30)

model.append('SGDClassifier')
precision.append(precision_score(y_test,y_pred))
recall.append(recall_score(y_test,y_pred))
F1score.append(f1_score(y_test,y_pred))
Accuracy.append(accuracy_score(y_test, y_pred))
AUCROC.append(roc_auc_score(y_test, y_prob))


"""** XGBoost **"""

from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(X_train,y_train)
y_pred = xgbc.predict(X_test)
y_prob=xgbc.predict_proba(X_test)[:,1]

print('='*60)
print("\nPerformance Metrics for XGBoost\n")
print('='*60)
print('Confusion Matrix')
print(confusion_matrix(y_test,y_pred),"\n")
print('-'*30)
print('Classification Report')
print(classification_report(y_test,y_pred),"\n")
print('-'*30)
print('AUC-ROC')
print(roc_auc_score(y_test, y_prob))
print('-'*30)

model.append('XGBoost')
recall.append(recall_score(y_test,y_pred))
precision.append(precision_score(y_test,y_pred))
F1score.append(f1_score(y_test,y_pred))
Accuracy.append(accuracy_score(y_test, y_pred))
AUCROC.append(roc_auc_score(y_test, y_prob))

print(model)
print(recall)
print(precision)
print(F1score)
print(Accuracy)

"""# Outlier removal: In the end to analyse

On closer inspection, we can see that the Rainfall,WindGustSpeed, Humidity3pm columns may contain outliers.

I will draw boxplots to visualise outliers in the above variables.
"""

# draw boxplots to visualize outliers

plt.figure(figsize=(15,10))

plt.subplot(3, 2, 1)
fig = df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')

plt.subplot(3, 2, 2)
fig = df.boxplot(column='WindGustSpeed')
fig.set_title('')
fig.set_ylabel('WindGustSpeed')

plt.subplot(3, 2, 5)
fig = df.boxplot(column='Humidity3pm')
fig.set_title('')
fig.set_ylabel('Humidity3pm')

column = ['Rainfall','WindGustSpeed','Humidity3pm']

def Outlier_detection(df, column):
    for i in column:
        IQR = df[i].quantile(0.75) - df[i].quantile(0.25)
        l_b = df[i].quantile(0.25) - (IQR * 3)
        u_b = df[i].quantile(0.75) + (IQR * 3)
        
        med = np.median(df[i])
        
        df[i] = np.where(df[i] > u_b , med,
                         np.where(df[i] < l_b, med, df[i]))
Outlier_detection(df, column)

df.describe()

df.columns