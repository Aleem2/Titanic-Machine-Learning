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
