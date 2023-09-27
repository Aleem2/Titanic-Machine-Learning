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
