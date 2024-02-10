# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 22:05:30 2024

@author: lEO
"""

# Import Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from buildml import SupervisedLearning

# G E T   D A T A S E T 
training_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

dataset = pd.concat([training_data, test_data], axis = 0)

# BuildML on Dataset
automate_training = SupervisedLearning(training_data)
automate_test = SupervisedLearning(test_data)

automate = [automate_training, automate_test]

# E X P L O R A T O R Y   D A T A   A N A L Y S I S 
training_eda = automate_training.eda()
test_eda = automate_test.eda()

# D A T A   C L E A N I N G   A N D   T R A N S F O R M A T I O N 
training_eda_visual = automate_training.eda_visual(figsize_barchart = (55, 10), figsize_heatmap = (15, 10), figsize_histogram=(35, 20))

for data in automate:
    data.reduce_data_memory_useage()
    data.drop_columns(["ID", "Batch Enrolled"])
    data.categorical_to_numerical()

drop_column = automate_test.drop_columns("Loan Status")
select_variables = automate_training.select_dependent_and_independent(predict = "Loan Status")
    
# To train this model, we would use the training data as the main data for this project. It will be split into further train and test sets.

# F U R T H E R   D A T A   P R E P A R A T I O N   A N D   S E G R E G A T I O N
training_data_clean = automate_training.get_dataset()
test_data_clean = automate_test.get_dataset()

unbalanced_dataset_check = automate_training.count_column_categories(column = "Loan Status")
split_data = automate_training.split_data()
# fix_unbalanced_data = automate_training.fix_unbalanced_dataset(sampler = "RandomOverSampler", random_state = 0)

check_unbalanced_data_fix = automate_training.count_column_categories(column = "Loan Status", test_data = True)

# M O D E L   B U I L D I N G 
classifiers = [LogisticRegression(random_state = 0),
                RandomForestClassifier(random_state = 0),
                DecisionTreeClassifier(random_state = 0),
                XGBClassifier(random_state = 0)
                ]

build_model = automate_training.build_multiple_classifiers(classifiers = classifiers,
                                                            kfold = 10,
                                                            cross_validation = True,
                                                            graph = True
                                                            )

# model = automate_training.train_model_classifier(classifiers[1])
# prediction = automate_training.classifier_predict() 
# evaluation = automate_training.classifier_evaluation(cross_validation = True)

