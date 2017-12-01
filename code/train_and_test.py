#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 14:40:03 2017

@author: songrenjie
"""

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from mlutils import utils
from mlutils import params_def
from mlutils import stacking
from sklearn.model_selection import cross_val_score

train = pd.read_csv('../input/fined_train.csv')
test = pd.read_csv('../input/fined_test.csv')
PassengerId = test['PassengerId']
train = train.drop('PassengerId', axis = 1)
test = test.drop('PassengerId', axis = 1)

y_train = train['Survived']
x_train = train.drop('Survived', axis = 1)

# Adaboost
DTC = DecisionTreeClassifier()
adaDTC = AdaBoostClassifier(DTC, random_state = 1024)
ada_best = utils.gs_train_model_or_load(adaDTC, params_def.ada_param_grid, 5,
                                        '../model/ada.m', x_train, y_train)
#ExtraTrees
ExtC = ExtraTreesClassifier()
## Search grid for optimal parameters
ExtC_best = utils.gs_train_model_or_load(ExtC, params_def.ex_param_grid, 5,
                                    '../model/ext.m', x_train, y_train)

# RFC Parameters tunning 
RFC = RandomForestClassifier()
## Search grid for optimal parameters
RFC_best = utils.gs_train_model_or_load(RFC, params_def.rf_param_grid, 5,
                                    '../model/rf.m', x_train, y_train)

# Gradient boosting tunning
GBC = GradientBoostingClassifier()
GBC_best = utils.gs_train_model_or_load(GBC, params_def.gb_param_grid, 5,
                                    '../model/gb.m', x_train, y_train)

### SVC classifier
SVMC = SVC(probability=True)
SVMC_best = utils.gs_train_model_or_load(SVMC, params_def.svc_param_grid, 5,
                                    '../model/svm.m', x_train, y_train)

votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
('svc', SVMC_best), ('gbc',GBC_best), ('adac', ada_best)],
voting='soft', n_jobs = -1)

votingC = votingC.fit(x_train, y_train)
test_Survived = pd.Series(votingC.predict(test), name = "Survived")
results = pd.concat([PassengerId, test_Survived], axis = 1)
results.to_csv("../output/submission.csv", index = False)

scores = cross_val_score(votingC, x_train, y_train, n_jobs = -1, cv = 5)
print("Accuracy: %0.5f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# =============================================================================
# validate_pre = votingC.predict(validate_fea)
# print ('accuracy on validate set: ', accuracy_score(validate_label,
#                                                     validate_pre))
# =============================================================================
# =============================================================================
# base_models = [ada_best, ExtC_best, RFC_best, GBC_best, SVMC_best]
# stacking_helper = stacking.StackingHelper()
# stacking_helper.add_base_models(base_models)
# gbm = xgb.XGBClassifier(**params_def.xgb_params)
# stacking_helper.add_decider(gbm)
# 
# y_train = train['Survived'].ravel()
# x_train = train.drop('Survived', axis = 1).values
# x_test = test.values
# 
# predictions = stacking_helper.train_and_predict(x_train, y_train, x_test)
# 
# print ('Training Complete!')
# StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
#                         'Survived': predictions })
# StackingSubmission.to_csv("../output/submission.csv", index=False)
# =============================================================================
