"""
Surprisingly, I have only heard about the Light Gradient Boosting Machine (LGBM) two days ago 
when I was working on a Machine Learning project with a friend. I immediately went online and 
did some research about it and found an article describing how the LGBM is the new XGBoost in
the sense that this is now the ultimate weapon of Data Scientists on Kaggle competitions as
well as in the industry. The main advantages of the LGBM to XGBoost are its significantly faster
processing speed in model building. And this was found to be true. The LGBM creators also
highlighted LGBM's improved accuracy over XGBoost but in this case this was not found to hold
true. 

As I have mentioned previously, the next stage is to implement LSTM Neural Network on the stock
dataset to conclude the project. But I felt that it was worth my time to try out the LGBM
considering that I have been dealing with much larger datasets lately for both Kaggle competitions
and my other Data Science projects.
"""

""" First import all the required libraries """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import fbeta_score, make_scorer

""" Import the required datasets """
SP500 = pd.read_csv('SP500_2.csv', parse_dates=True)
Nasdaq = pd.read_csv('Nasdaq_2.csv', parse_dates=True)
DJI = pd.read_csv('DJI_2.csv', parse_dates=True)
DAX = pd.read_csv('DAX_2.csv', parse_dates=True)
Paris = pd.read_csv('Paris_2.csv', parse_dates=True)
Tokyo = pd.read_csv('Tokyo_2.csv', parse_dates=True)
HongKong = pd.read_csv('HongKong_2.csv', parse_dates=True)
Aus = pd.read_csv('Aus_2.csv', parse_dates=True)

"""
Since we had previously exported our engineered features as a CSV file,
we can just import it in instead of going through the feature engineering
procedure with our own functions again
"""
reduced_megaset = pd.read_csv('reduced_megaset.csv')

""" The target variable is the daily return of each day, binary encoded """
target_raw = (SP500['Adj Close'].shift(-1)/SP500['Adj Close'])-1
target = target_raw[21:]
target[target > 0] = 1
target[target <= 0] = 0

"""
Split our dimension_reduced megaset and the target array into training and testing
subsets
"""
X_train = reduced_megaset[:6001]
X_test = reduced_megaset[6001:-1]
y_train = target[:6001]
y_test = target[6001:-1]

""" LightGBM Implementation and Results"""
# We need to convert our training data into LightGBM dataset format
d_train = lgb.Dataset(X_train, label=y_train)
# Setting parameters for training
# objective set to binary for binary classification problem
# boosting_type set to gbdt for gradient boosting
# binary_logloss as metric for binary classification predictions
# other parameters randomly selected and subject to change for optimization
params = {'boosting_type': 'gbdt',
          'learning_rate': 0.003,
          'max_depth': 10,
          'metric': 'binary_logloss',
          'min_data': 50,
          'num_leaves': 10,
          'objective': 'binary',
          'sub_feature': 0.5}
# fit the clf_LGBM on training data with 100 training iterations
clf_LGBM = lgb.train(params, d_train, 100)
# make predictions with test data
y_pred = clf_LGBM.predict(X_test)
# sinec the output is a list of probabilities, below we have converted the probabilities
# to binary prediction with threshold set at 0.5
for i in range(0, len(y_pred)):
    if y_pred[i] >= 0.5:
       y_pred[i] = 1
    else:  
       y_pred[i]=0
# evaluate predictions with accuracy metric
clf_LGBM_accuracy = accuracy_score(y_test, y_pred)
# evaluate predictions with F1-score metric
clf_LGBM_f1 = f1_score(y_test, y_pred)
print("LightGBM Classifier [Accuracy score: {:.4f}, f1-score: {:.4f}]".format(clf_LGBM_accuracy, clf_LGBM_f1))
# LightGBM Classifier [Accuracy score: 0.5783, f1-score: 0.7155]

""" Optimization of LightGBM """
# Choose LGBM Classifier as the algorithm for optimization with GridSearch
clf_LGBM2 = lgb.LGBMClassifier(boosting_type = 'gbdt', metric = 'binary_logloss', 
                               min_data = 50, objective = 'binary', sub_feature = 0.5)
# Create a dictionary for the parameters
gridParams = {'learning_rate': [0.0001, 0.0003, 0.0005, 0.001],'n_estimators': [75, 100, 125],
             'num_leaves': [15, 16, 17],'colsample_bytree' : [0.58, 0.60, 0.62],'subsample' : [0.4, 0.5, 0.7]}
# Choose the time series cross-validator
tscv = TimeSeriesSplit(n_splits=3)
# Create the GridSearch object
grid = GridSearchCV(clf_LGBM2, gridParams, verbose=1, cv= tscv)
# Fit the grid search object to the data to compute the optimal model
grid_fit_LGBM = grid.fit(X_train, y_train)
# Return the optimal model after fitting the data
best_clf_LGBM = grid_fit_LGBM.best_estimator_
# Make predictions with the optimal model
best_predictions_LGBM = best_clf_LGBM.predict(X_test)
# Get the accuracy and F1_score of the optimized model
clf_LGBM_optimized_accuracy = accuracy_score(y_test, best_predictions_LGBM)
clf_LGBM_optimized_f1 = f1_score(y_test, best_predictions_LGBM)
print("LGBM Classifier Optimized [Accuracy score: {:.4f}, f1-score: {:.4f}]".format(clf_LGBM_optimized_accuracy, clf_LGBM_optimized_f1))
print(grid_fit_LGBM.best_params_)
print(grid_fit_LGBM.best_score_)
# Output:
# LGBM Classifier Optimized [Accuracy score: 0.5739, f1-score: 0.7263]
# {'colsample_bytree': 0.58, 'learning_rate': 0.001, 'n_estimators': 75, 'num_leaves': 17, 'subsample': 0.5}
# 0.532
# [Parallel(n_jobs=1)]: Done 972 out of 972 | elapsed:  1.2min finished