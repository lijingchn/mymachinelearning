#!/usr/bin/env python
# encoding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

train_df = pd.read_csv("../data/train.csv", dtype={"Age": np.float64})

