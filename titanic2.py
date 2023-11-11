# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
train = pd.read_csv("train.csv")
print(train.head())
print(train.describe())
print(train.head())
print("getting header")
print(train.columns)
print(train['Name'])
print(train.iloc[:4,:3])
print(train.iloc[3,:])
print(train.iloc[4,:3])
print(train.iloc[:,2])
print(train.loc[:,['Pclass','Age']])
# Suggestion to my self is stick to loc and iloc to refer to specific column and rows.
print(train.loc[:,['PassengerId','Survived']])
# Now I want to see stats of of the survived column. How can I do so?
# One approach is to create a sub data frame and run describe on it. Infact without creating a variable. 
print((train.loc[:,['PassengerId','Survived']]).describe())
#testing out the fetch git command.
#tested successfully.

#Collected and clean subset of the data for training
train_data=train.loc[:,['PassengerId','Sex','Age','Pclass','Survived']]
train_data['Age'].interpolate(str='linear',inplace=True)
train_data.describe()
test=pd.read_csv('test.csv')
test.describe()
test_data=test.loc[:,['PassengerId','Sex','Age','Pclass']]
# to note that Survived column is missing
test_data['Age'].interpolate(str='linear',inplace=True)
test_data.describe()
#-----------------------------------
#Using Scikit Learn to train models

#Creating the training dataset
# feature matrix
X = train_data[['PassengerId','Sex','Age','Pclass']]
X.loc[:,'Sex'] = X.loc[:,'Sex'].apply(lambda x: 1 if x == 'male' else 0)
# result matrix
y = train_data['Survived']
y.describe()

#

# example code from tutorial https://scikit-learn.org/stable/getting_started.html
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# create a pipeline object
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)
# load the iris dataset and split it into train and test sets
#X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
#print(y_train)
# fit the whole pipeline
pipe.fit(X_train, y_train)
# we can now use it like any other estimator
print(accuracy_score(pipe.predict(X_test), y_test))

#----------------------------------------------------------------------------------------
#Model evaluation - cross validation by splitting the data using different approaches.
from sklearn.model_selection import cross_validate
result = cross_validate(pipe, X, y)
print(result['test_score'])


print(pipe.get_params)
