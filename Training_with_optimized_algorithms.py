import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score, make_scorer

"""Because this is time series data, we cannot do shuffle split.
   Instead, we are going to use the TimeSeriesSplit function
   provided in sklearn"""

"""Optimization of SVC"""
# Choose the time series cross-validator
tscv = TimeSeriesSplit(n_splits=10)
# Choose SVC as the algorithm for optimization with GridSearch
clf8 = svm.SVC()
# Create a dictionary for the parameters
parameters_SVC = {'kernel':('linear', 'rbf', 'poly'), 'C':[1, 100, 200], 'degree':[200, 400, 600, 800]}
# Define a scoring function
scorer = make_scorer(f1_score)
# Create the GridSearch object"""
grid_obj_SVC = GridSearchCV(estimator=clf8, param_grid=parameters_SVC, scoring=scorer, cv=tscv)
# Fit the grid search object to the data to compute the optimal model
grid_fit_SVC = grid_obj_SVC.fit(X_train, y_train)
# Return the optimal model after fitting the data
best_clf_SVC = grid_fit_SVC.best_estimator_
# Make predictions with the optimal model
best_predictions_SVC = best_clf_SVC.predict(X_test)
# Get the accuracy and f1_score of the optimized model
clf8_optimized_accuracy = accuracy_score(y_test, best_predictions_SVC)
clf8_optimized_f1 = f1_score(y_test, best_predictions_SVC)
print("SVC Optimized [Accuracy score: {:.4f}, f1-score: {:.4f}]".format(clf8_optimized_accuracy, clf8_optimized_f1))

"""Optimization of decision tree"""
# Choose DT as the algorithm for optimization with GridSearch
clf9 = tree.DecisionTreeClassifier()
# Create a dictionary for the parameters
parameters_DT = {'criterion':('gini', 'entropy')}
# Define a scoring function
scorer = make_scorer(f1_score)
# Create the GridSearch object"""
grid_obj_DT = GridSearchCV(estimator=clf9, param_grid=parameters_DT, scoring=scorer, cv=tscv)
# Fit the grid search object to the data to compute the optimal model
grid_fit_DT = grid_obj_DT.fit(X_train, y_train)
# Return the optimal model after fitting the data
best_clf_DT = grid_fit_DT.best_estimator_
# Make predictions with the optimal model
best_predictions_DT = best_clf_DT.predict(X_test)
# Get the accuracy and f1_score of the optimized model
clf9_optimized_accuracy = accuracy_score(y_test, best_predictions_DT)
clf9_optimized_f1 = f1_score(y_test, best_predictions_DT)
print("Decidion Tree Optimized [Accuracy score: {:.4f}, f1-score: {:.4f}]".format(clf9_optimized_accuracy, clf9_optimized_f1))

"""Optimization of random forest"""
# Choose RF as the algorithm for optimization with GridSearch
clf10 = RandomForestClassifier()
# Create a dictionary for the parameters
parameters_RF = {'n_estimators':[5, 10, 20], 'criterion':('gini', 'entropy')}
# Define a scoring function
scorer = make_scorer(f1_score)
# Create the GridSearch object"""
grid_obj_RF = GridSearchCV(estimator=clf10, param_grid=parameters_RF, scoring=scorer, cv=tscv)
# Fit the grid search object to the data to compute the optimal model
grid_fit_RF = grid_obj_RF.fit(X_train, y_train)
# Return the optimal model after fitting the data
best_clf_RF = grid_fit_RF.best_estimator_
# Make predictions with the optimal model
best_predictions_RF = best_clf_RF.predict(X_test)
# Get the accuracy and f1_score of the optimized model
clf10_optimized_accuracy = accuracy_score(y_test, best_predictions_RF)
clf10_optimized_f1 = f1_score(y_test, best_predictions_RF)
print("Decidion Tree Optimized [Accuracy score: {:.4f}, f1-score: {:.4f}]".format(clf10_optimized_accuracy, clf10_optimized_f1))

"""Optimization of kNN"""
# Choose the time series cross-validator
tscv = TimeSeriesSplit(n_splits=3)
# Choose SVC as the algorithm for optimization with GridSearch
clf11 = KNeighborsClassifier()
# Create a dictionary for the parameters
parameters_kNN = {'weights':('uniform', 'distance'), 'n_neighbors':[3, 5, 7, 10, 25, 50, 100]}
# Define a scoring function
scorer = make_scorer(f1_score)
# Create the GridSearch object"""
grid_obj_kNN = GridSearchCV(estimator=clf11, param_grid=parameters_kNN, scoring=scorer, cv=tscv)
# Fit the grid search object to the data to compute the optimal model
grid_fit_kNN = grid_obj_kNN.fit(X_train, y_train)
# Return the optimal model after fitting the data
best_clf_kNN = grid_fit_kNN.best_estimator_
# Make predictions with the optimal model
best_predictions_kNN = best_clf_kNN.predict(X_test)
# Get the accuracy and f1_score of the optimized model
clf11_optimized_accuracy = accuracy_score(y_test, best_predictions_kNN)
clf11_optimized_f1 = f1_score(y_test, best_predictions_kNN)
print("kNN Optimized [Accuracy score: {:.4f}, f1-score: {:.4f}]".format(clf11_optimized_accuracy, clf11_optimized_f1))

"""Optimization of AdaBoost with DecisionTree"""
# Choose the time series cross-validator
tscv = TimeSeriesSplit(n_splits=3)
# Choose SVC as the algorithm for optimization with GridSearch
clf12 = AdaBoostClassifier(base_estimator = DecisionTreeClassifier())
# Create a dictionary for the parameters
parameters_AdaBoost1 = {'n_estimators':[3, 5, 7, 10, 25, 50, 75, 100], 'learning_rate':[2, 3, 4, 5, 6]}
# Define a scoring function
scorer = make_scorer(f1_score)
# Create the GridSearch object"""
grid_obj_AdaBoost1 = GridSearchCV(estimator=clf12, param_grid=parameters_AdaBoost1, scoring=scorer, cv=tscv)
# Fit the grid search object to the data to compute the optimal model
grid_fit_AdaBoost1 = grid_obj_AdaBoost1.fit(X_train, y_train)
# Return the optimal model after fitting the data
best_clf_AdaBoost1 = grid_fit_AdaBoost1.best_estimator_
# Make predictions with the optimal model
best_predictions_AdaBoost1 = best_clf_AdaBoost1.predict(X_test)
# Get the accuracy and f1_score of the optimized model
clf12_optimized_accuracy = accuracy_score(y_test, best_predictions_AdaBoost1)
clf12_optimized_f1 = f1_score(y_test, best_predictions_AdaBoost1)
print("AdaBoost DT Optimized [Accuracy score: {:.4f}, f1-score: {:.4f}]".format(clf12_optimized_accuracy, clf12_optimized_f1))

"""Optimization of AdaBoost with GaussianNB"""
# Choose the time series cross-validator
tscv = TimeSeriesSplit(n_splits=3)
# Choose SVC as the algorithm for optimization with GridSearch
clf13 = AdaBoostClassifier(base_estimator = GaussianNB())
# Create a dictionary for the parameters
parameters_AdaBoost2 = {'n_estimators':[3, 5, 7, 10, 25, 50, 75, 100], 'learning_rate':[2, 3, 4, 5, 6]}
# Define a scoring function
scorer = make_scorer(f1_score)
# Create the GridSearch object"""
grid_obj_AdaBoost2 = GridSearchCV(estimator=clf13, param_grid=parameters_AdaBoost2, scoring=scorer, cv=tscv)
# Fit the grid search object to the data to compute the optimal model
grid_fit_AdaBoost2 = grid_obj_AdaBoost2.fit(X_train, y_train)
# Return the optimal model after fitting the data
best_clf_AdaBoost2 = grid_fit_AdaBoost2.best_estimator_
# Make predictions with the optimal model
best_predictions_AdaBoost2 = best_clf_AdaBoost2.predict(X_test)
# Get the accuracy and f1_score of the optimized model
clf13_optimized_accuracy = accuracy_score(y_test, best_predictions_AdaBoost2)
clf13_optimized_f1 = f1_score(y_test, best_predictions_AdaBoost2)
print("AdaBoost DT Optimized [Accuracy score: {:.4f}, f1-score: {:.4f}]".format(clf13_optimized_accuracy, clf13_optimized_f1))