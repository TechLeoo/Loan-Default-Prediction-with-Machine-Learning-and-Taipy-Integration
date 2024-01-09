# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 22:05:30 2024

@author: lEO
"""

# Import Libraries
import pandas as pd
import numpy as np
import joblib
from buildml.automate import SupervisedLearning

# Get Dataset
training_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

dataset = pd.concat([training_data, test_data], axis = 0)

# BuildML on Dataset
automate_training = SupervisedLearning(training_data)
automate_test = SupervisedLearning(test_data)

automate = [automate_training, automate_test]

# Exploratory Data Analysis
training_eda = automate_training.eda()
test_eda = automate_test.eda()

# Data Cleaning and Transformation
training_eda_visual = automate_training.eda_visual(y = "Loan Status", figsize_barchart = (55, 10), figsize_heatmap = (15, 10), figsize_histogram=(35, 20))

for data in automate:
    data.drop_columns(["ID", "Batch Enrolled"])
    data.categorical_to_numerical()
    data.select_dependent_and_independent(predict = "Loan Status")
    clean = data.get_dataset()
    
    

