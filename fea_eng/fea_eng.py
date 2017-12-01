#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:37:56 2017

@author: songrenjie
"""

# Load in our libraries
import pandas as pd
import numpy as np
import re
import random
import warnings
from sklearn import linear_model
from mlutils import utils
from mlutils import params_def
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# Load in the train and test datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
full_data = [train, test]

# Store our passenger ID for easy access
PassengerId = test['PassengerId']

# Remove all NULLS in the Embarked column
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# Remove all NULLS in the Fare column and create a new feature CategoricalFare
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Create a new feature Title, containing the titles of passenger names
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt',
           'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
    'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
for dataset in full_data:
    # New features about family size
# =============================================================================
#     dataset['IsSingleWomen'] = 1
#     dataset.loc[(dataset['Parch'] + dataset['SibSp'] > 0) *
#                 (dataset['Sex'] == 'female'), 'IsSingleWomen'] = 0
#     dataset['IsSingleMen'] = 0
#     dataset.loc[(dataset['Parch'] + dataset['SibSp'] > 0) *
#                 (dataset['Sex'] == 'male'), 'IsSingleMen'] = 1
# =============================================================================
# =============================================================================
#     dataset['IsSingle'] = (dataset['Parch'] +
#            dataset['SibSp'] == 0).astype(int)
# =============================================================================
# =============================================================================
#     dataset['Fsize'] = dataset['Parch'] + dataset['SibSp'] + 1
# =============================================================================
# =============================================================================
#     dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
# =============================================================================
# =============================================================================
#     dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)
#     dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
#     dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)
# =============================================================================
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map(
            {'S': 0, 'C': 1, 'Q': 2}).astype(int)
    
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping Fare
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454),
                'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31),
                'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
# =============================================================================
#     dataset['HasCabin'] = (dataset['Cabin'].notnull()).astype(int)
# =============================================================================

    # Mapping Age
# =============================================================================
#     dataset['HasAge'] = dataset['Age'].notnull().astype(int)
# =============================================================================
# =============================================================================
#     age_avg = dataset['Age'].mean()
#     age_std = dataset['Age'].std()
#     age_null_count = dataset['Age'].isnull().sum()
#     age_null_random_list = np.random.randint(age_avg - age_std,
#                                              age_avg + age_std,
#                                              size=age_null_count)
#     dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
#     dataset['Age'] = dataset['Age'].astype(int)
# =============================================================================
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
# =============================================================================
#     dataset.loc[(dataset['Age'] > 8) & (dataset['Age'] <= 16), 'Age'] = 1
# =============================================================================
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 64), 'Age'] = 2
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 3
    rf = RandomForestClassifier()
    dataset = utils.clf_fillna(rf,
                               dataset,
                               ['Title', 'Sex', 'Pclass', 'Parch'],
                               'Age')
    dataset['Age'] = dataset['Age'].astype(int)
    
# =============================================================================
#     dataset['P1Women'] = ((dataset['Pclass']) == 1 &
#            (dataset['Sex'] == 0)).astype(int)
#     dataset['P1Men'] = ((dataset['Pclass']) == 1 &
#            (dataset['Sex'] == 1)).astype(int)
#     dataset['P2Women'] = ((dataset['Pclass']) == 2 &
#            (dataset['Sex'] == 0)).astype(int)
#     dataset['P2Men'] = ((dataset['Pclass']) == 2 &
#            (dataset['Sex'] == 1)).astype(int)
#     dataset['P3Women'] = ((dataset['Pclass']) == 3 &
#            (dataset['Sex'] == 0)).astype(int)
#     dataset['P3Men'] = ((dataset['Pclass']) == 3 &
#            (dataset['Sex'] == 1)).astype(int)
# =============================================================================
    
drop_features = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
train = train.drop(drop_features, axis = 1)
test = test.drop(drop_features, axis = 1)

print (train.head(0))
print ('*' * 40)
print (test.head(0))


# =============================================================================
# validate_idx = range(int(train.shape[0] * 0.2), 2 * int(train.shape[0] * 0.2))
# validate_set = train.iloc[validate_idx]
# validate_set.to_csv('../input/validate.csv', index = False)
# train = train.drop(validate_idx, axis = 0)
# =============================================================================
train.to_csv("../input/fined_train.csv", index = False)
test.to_csv("../input/fined_test.csv", index = False)