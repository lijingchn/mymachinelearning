#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
import re
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import pointbiserialr,spearmanr
from sklearn.cross_validation import train_test_split

train_df = pd.read_csv("../data/train.csv", dtype={"Age": np.float64})
test_df = pd.read_csv("../data/test.csv", dtype={"Age": np.float64})

title_lst = []
for i in train_df["Name"]:
    matcher = re.match(r"^\D+,\s(.*?\.)\s\D+",i)
    title_lst.append(matcher.group(1))
#print list(set(title_lst))

Title_Dictionary = {
                "Capt": "Officer",
                "Col":  "Officer",
                "Major":"Officer",
                "Jonkheer":"Royalty",
                "Don":  "Royalty",
                "Sir":  "Royalty",
                "Dr":   "Officer",
                "Rev":  "Officer",
                "the Countess": "Royalty",
                "Dona":  "Royalty",
                "Mme":   "Mrs",
                "Mlle":  "Miss",
                "Ms":    "Mrs",
                "Mr":    "Mr",
                "Mrs":   "Mrs",
                "Miss":  "Miss",
                "Master":"Master",
                "Lady":  "Royalty"
}

# process data
train_df["Title"] = train_df["Name"].apply(lambda x:Title_Dictionary[x.split(',')[1].split('.')[0].strip()])

# fill the missingness data
train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
train_df["Fare"] = train_df["Fare"].fillna(train_df["Fare"].median())
train_df["Embarked"] = train_df["Embarked"].fillna("S")
#print train_df["Embarked"][train_df["Embarked"].isnull()]

#train_df["Sex"][train_df["Sex"] == "female"] = 0
#train_df["Sex"][train_df["Sex"] == "male"] = 1
#train_df["Embarked"][train_df["Embarked"] == "S"] = 0
#train_df["Embarked"][train_df["Embarked"] == "C"] = 1
#train_df["Embarked"][train_df["Embarked"] == "Q"] = 2
train_df["Family_size"] = train_df["Parch"] + train_df["SibSp"] + 1

# drop useless series
train_df = train_df.drop(['Ticket','Cabin','SibSp','Parch'], axis = 1)
#train_df.hist(figsize=(20,15))
#plt.show()

dummies_Sex = pd.get_dummies(train_df["Sex"], prefix = 'Sex')
dummies_Embarked = pd.get_dummies(train_df["Embarked"], prefix = 'Embarked')
dummies_Pclass = pd.get_dummies(train_df["Pclass"], prefix = 'Pclass')
dummies_Title = pd.get_dummies(train_df["Title"], prefix = 'Title')
train_df = pd.concat([train_df,dummies_Sex,dummies_Embarked,dummies_Pclass,dummies_Title], axis=1)
train_df = train_df.drop(["Sex","Embarked","Pclass","Title","Name"], axis=1)
train_df.set_index(['PassengerId'])


columns = train_df.columns.values
param = []
correlation = []
abs_corr = []

for c in columns:
    # Check if binary or continuous
    if len(train_df[c].unique()) <= 2:
        corr = spearmanr(train_df["Survived"], train_df[c])[0]
    else:
        corr = pointbiserialr(train_df["Survived"], train_df[c])[0]
    param.append(c)
    correlation.append(corr)
    abs_corr.append(abs(corr))

# Create dataframe for visualization
param_df = pd.DataFrame({"correlation":correlation, "parameter":param, "abs_corr":abs_corr})
# Sort by absolute correlation
param_df = param_df.sort_values(by=["abs_corr"],ascending=False)
# Set parameter name as index
param_df = param_df.set_index("parameter")
print param_df

scoresCV = []
scores = []
print '================='
print len(param_df)

for i in range(1,len(param_df)):
    new_df = train_df[param_df.index[0:i+1].values]
    X = new_df.ix[:,1::]
    y = new_df.ix[:,0]
    clf = DecisionTreeClassifier()
    scoreCV = sklearn.cross_validation.cross_val_score(clf,X,y,cv=10)
    scores.append(np.mean(scoreCV))
    print scores
    print len(scores)

plt.figure(figsize = (15,5))
plt.plot(range(1,len(scores)+1), scores, '.-')
#plt.axis("tight")
plt.title("Feature Selection", fontsize=14)
plt.xlabel("# Features", fontsize=12)
plt.ylabel("Score", fontsize=12)
#plt.show()

# Chose the best features
best_features = param_df.index[1:10+1].values
print best_features
train_df[best_features].hist(figsize=(20,15))
#plt.show()

# Creating the train and test datasets
X = train_df[best_features]
y = train_df["Survived"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=44)

# decision tree
plt.figure(figsize=(15,7))

# Max Features
plt.subplot(2,3,1)
feature_param = ["auto","sqrt","log2",None]
scores = []



# Building and fitting my forest 
target = train_df["Survived"].values
#features_forest = train_df[["Family_size","Pclass","Sex","Age","Fare","SibSp", "Parch", "Embarked"]].values
#features_forest = train_df[["Pclass","Sex","Age","Fare","SibSp", "Parch", "Embarked"]].values
#forest = RandomForestClassifier(n_estimators=100, n_jobs=4, max_features="sqrt")
#my_forest = forest.fit(features_forest,target)
#print "======================================"
#print my_forest.score(features_forest,target)

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

#test_features = test_df[["Family_size","Pclass","Sex","Age","Fare","SibSp", "Parch", "Embarked"]].values
#test_features = test_df[["Pclass","Sex","Age","Fare","SibSp", "Parch", "Embarked"]].values
#my_prediction = my_forest.predict(test_features)

# Create a data frame with two columns: PassengerId & Survived.
#PassengerId = np.array(test_df["PassengerId"]).astype(int)
#my_solution = pd.DataFrame(my_prediction, PassengerId, columns=["Survived"])
#print my_solution
#print my_solution.shape

# Write solution to a csv file with the name my_solution.csv
#my_solution.to_csv("my_forest_solution.csv", index_label=["PassengerId"])

