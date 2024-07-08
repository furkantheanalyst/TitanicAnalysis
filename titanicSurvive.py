# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 13:08:36 2021

@author: Furkan
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("train.csv")
pd.set_option('display.max_columns', None)
print(df.head())
print(df.isnull().sum().sort_values(ascending=False))


#Visualizing Data
sns.set_style("white")
ax = sns.countplot(x="Survived", data=df)
plt.show()

sns.set_style("white")
ax = sns.countplot(x="Survived", hue="Sex", data=df)
plt.show()

sns.histplot(data=df, x="Age", binwidth=3)
plt.show()

sns.set_style("white")
ax=sns.countplot(x='Embarked',data=df)
plt.show()

sns.set_style("white")
ax=sns.countplot(x='Pclass',data=df)
plt.title('Class')
plt.show()

sns.set_style("white")
ax=sns.countplot(x='Parch',data=df)
plt.title("Number of Parents/Children Aboard")
plt.show()

sns.set_style("white")
ax=sns.countplot(x='SibSp',data=df)
plt.title("Number of Siblings/Spouses Aboard")
plt.show()

sns.histplot(data=df, y="Fare", binwidth=3)
plt.show()

#Data Cleaning
df.drop(["Cabin","Name","Ticket"], axis=1, inplace=True)
#Most of the passengers are came from Southampton, so i accept that as a mean.
df["Embarked"].fillna("S",inplace=True)
print("\n",df.head())

#Converting Categorical Variables To Pandas Dummies
df = pd.get_dummies(df, columns=['Sex','Embarked'])
df.drop(["Sex_female","Embarked_C"],axis=1,inplace=True)

#Filling Na Ages With Class Means
print("\n",df.groupby(['Pclass']).mean())
df['Age'] = df['Age'].fillna(df.groupby('Pclass')['Age'].transform('mean'))

print("----------------------------------")
print(df.isnull().sum().sort_values(ascending=False))
print(df.head())
print(df.corr())

x=df.drop(['Survived'],axis=1)
y=df['Survived']
# ------------------------------------------------------
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#Scaling
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("LR")
print(cm)



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('KNN')
print(cm)



from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)



from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)


y_proba = rfc.predict_proba(X_test)
print(y_test)
#Survive percentage
print(y_proba[:,0])

