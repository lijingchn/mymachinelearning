#!/usr/bin/env python
# encoding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv("../data/train.csv", dtype={"Age": np.float64})
test_df = pd.read_csv("../data/test.csv", dtype={"Age": np.float64})

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
train_df["Family_size"] = train_df["Parch"] + train_df["SibSp"] + 1

# Building and fitting my forest
target = train_df["Survived"].values
features_forest = train_df[["Family_size","Pclass","Sex","Age","Fare","SibSp", "Parch", "Embarked"]].values
forest = RandomForestClassifier(n_estimators=100, n_jobs=4, max_features="sqrt")
my_forest = forest.fit(features_forest,target)
print "======================================"
print my_forest.score(features_forest,target)

#print train_df["Survived"][train_df["Child"] == 1].value_counts()
#print train_df["Child"][train_df["Child"]==1]
#print train_df[train_df["Child"]==1]

#Impute the missing value for Fare in row 152 with the median of the columns
#print test_df["Fare"][test_df["Fare"].isnull()]
test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].median())
test_df["Age"] = test_df["Age"].fillna(test_df["Fare"].median())
test_df["Sex"][test_df["Sex"] == "female"] = 0
test_df["Sex"][test_df["Sex"] == "male"] =1
test_df["Embarked"][test_df["Embarked"] == "S"] = 0
test_df["Embarked"][test_df["Embarked"] == "C"] = 1
test_df["Embarked"][test_df["Embarked"] == "Q"] = 2
test_df["Family_size"] = test_df["Parch"] + test_df["SibSp"] + 1

test_features = test_df[["Family_size","Pclass","Sex","Age","Fare","SibSp", "Parch", "Embarked"]].values
my_prediction = my_forest.predict(test_features)

# Create a data frame with two columns: PassengerId & Survived.
PassengerId = np.array(test_df["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns=["Survived"])
#print my_solution
#print my_solution.shape

# Write solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_forest_solution.csv", index_label=["PassengerId"])

