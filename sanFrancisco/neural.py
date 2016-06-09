#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA
from copy import deepcopy

# Import data
train_df = pd.read_csv("data/train.csv")
print train_df[["X","Y"]].head(100)

# Clean up wrong X and Y values (very few of them)
xy_scaler = preprocessing.StandardScaler()
xy_scaler.fit(train_df[["X","Y"]])
train_df[["X","Y"]] = xy_scaler.transform(train_df[["X","Y"]])
train_df = train_df[abs(train_df["Y"]) < 100]
train_df.index = range(len(train_df))
print train_df[["X","Y"]].head(100)
plt.plot(train_df["X"], train_df["Y"], '.')
plt.show()

