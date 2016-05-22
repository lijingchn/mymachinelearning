#!/usr/bin/env python
# encoding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

train = pd.read_csv("../data/train.csv", dtype={"Age": np.float64}, )

# Replacing missing ages with median
missAge = np.median(train["Age"].fillna(0))
print '==========='
print missAge

train["Age"][np.isnan(train["Age"])] = missAge
print train

train["Survived"][train["Survived"]==1] = "Survived"
train["Survived"][train["Survived"]==0] = "Died"
train["ParentsAndChildren"] = train["Parch"]
train["SiblingsAndSpouses"] = train["SibSp"]

plt.figure()
#sns.pairplot(data=train[["Fare","Survived","Age","ParentsAndChildren","SiblingsAndSpouses","Pclass"]],
#             hue="Survived", dropna=True)
sns.pairplot(data=train[["Survived","Age","ParentsAndChildren","SiblingsAndSpouses"]],
             hue="Survived", dropna=True)
#plt.savefig("../pic/1_seaborn_pair_plot.png")
plt.show()