#!/usr/bin/env python
# encoding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import tree

train_df = pd.read_csv("../data/train.csv", dtype={"Age": np.float64})
test_df = pd.read_csv("../data/test.csv", dtype={"Age": np.float64})

#train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
#test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
#train_df = pd.read_csv(train_url)

train_df["Child"] = 0
train_df["Child"][train_df["Age"]<18] = 1

# fill the missingness data
train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
train_df["Fare"] = train_df["Fare"].fillna(train_df["Fare"].median())
train_df["Embarked"] = train_df["Embarked"].fillna("S")
#print train_df["Embarked"][train_df["Embarked"].isnull()]

train_df["Sex"][train_df["Sex"] == "female"] = 0
train_df["Sex"][train_df["Sex"] == "male"] = 1
train_df["Embarked"][train_df["Embarked"] == "S"] = 0
train_df["Embarked"][train_df["Embarked"] == "C"] = 1
train_df["Embarked"][train_df["Embarked"] == "Q"] = 2
#print train_df
train_df["Family_size"] = train_df["Parch"] + train_df["SibSp"] + 1

# drow the picture
plt.figure()
sns.pairplot(data=train_df[["Family_size","Survived","Sex","Age"]],hue="Survived",dropna=True)
plt.show()

max_depth = 10
min_samples_split = 5
target = train_df["Survived"].values
features = train_df[["Family_size","Pclass","Sex","Age","Fare"]].values
my_tree = tree.DecisionTreeClassifier(max_depth=max_depth,min_samples_split=min_samples_split, random_state=1)
my_tree = my_tree.fit(features,target)
#print my_tree.feature_importances_
print my_tree.score(features,target)

#print train_df["Survived"][train_df["Child"] == 1].value_counts()
#print train_df["Child"][train_df["Child"]==1]
#print train_df[train_df["Child"]==1]

#Impute the missing value for Fare in row 152 with the median of the columns
#print test_df["Fare"][test_df["Fare"].isnull()]
test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].median())
test_df["Age"] = test_df["Age"].fillna(test_df["Fare"].median())
test_df["Sex"][test_df["Sex"] == "female"] = 0
test_df["Sex"][test_df["Sex"] == "male"] =1
test_df["Family_size"] = test_df["Parch"] + test_df["SibSp"] + 1
test_features = test_df[["Family_size","Pclass","Sex","Age","Fare"]].values
my_prediction = my_tree.predict(test_features)

# Create a data frame with two columns: PassengerId & Survived.
PassengerId = np.array(test_df["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns=["Survived"])
#print my_solution
#print my_solution.shape

# Write solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution.csv", index_label=["PassengerId"])

