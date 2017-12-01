#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:33:24 2017

@author: songrenjie
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
PassengerId = test['PassengerId']

sns.set(style='white', context='notebook', palette='deep')

train['Fare'] = train['Fare'].map(lambda x : np.log(x) if x > 0 else 0)
g = sns.distplot(train['Fare'], color = 'm', label = 'Skewness : %.2f'
                % (train['Fare'].skew()))
g = g.legend(loc = 'best')

g = sns.FacetGrid(train, col = 'Survived')
g = g.map(sns.distplot, 'Age')