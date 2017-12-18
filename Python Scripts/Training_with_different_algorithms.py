import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

"""Support Vector Classifier with Linear Kernel"""
clf1 = svm.SVC(kernel = 'linear')
clf1.fit(X_train, y_train)
clf1_predictions = clf1.predict(X_test)
clf1_accuracy = accuracy_score(y_test, clf1_predictions)
clf1_f1 = f1_score(y_test, clf1_predictions)
print("SVM Linear: [Accuracy: {:.4f}, f1-score: {:.4f}]".format(clf1_accuracy, clf1_f1))

"""Support Vector Classifier with RBF Kernel"""
clf2 = svm.SVC(kernel = 'rbf')
clf2.fit(X_train, y_train)
clf2_predictions = clf2.predict(X_test)
clf2_accuracy = accuracy_score(y_test, clf2_predictions)
clf2_f1 = f1_score(y_test, clf2_predictions)
print("SVM RBF: [Accuracy: {:.4f}, f1-score: {:.4f}]".format(clf2_accuracy, clf2_f1))

"""k-Nearest Neighbours"""
clf3 = KNeighborsClassifier(n_neighbors = 3)
clf3.fit(X_train, y_train)
clf3_predictions = clf3.predict(X_test)
clf3_accuracy = accuracy_score(y_test, clf3_predictions)
clf3_f1 = f1_score(y_test, clf3_predictions)
print("kNN: [Accuracy: {:.4f}, f1-score: {:.4f}]".format(clf3_accuracy, clf3_f1))

"""Decision Tree Classifier"""
clf4 = tree.DecisionTreeClassifier()
clf4.fit(X_train, y_train)
clf4_predictions = clf4.predict(X_test)
clf4_accuracy = accuracy_score(y_test, clf4_predictions)
clf4_f1 = f1_score(y_test, clf4_predictions)
print("Decision Tree: [Accuracy: {:.4f}, f1-score: {:.4f}]".format(clf4_accuracy, clf4_f1))

"""Random Forest Classifier"""
clf5 = RandomForestClassifier(n_estimators=10)
clf5.fit(X_train, y_train)
clf5_predictions = clf4.predict(X_test)
clf5_accuracy = accuracy_score(y_test, clf5_predictions)
clf5_f1 = f1_score(y_test, clf5_predictions)
print("Random Forest Classifier: [Accuracy: {:.4f}, f1-score: {:.4f}]".format(clf5_accuracy, clf5_f1))

"""AdaBoost Classifier with DecisionTree base"""
clf6a = AdaBoostClassifier(n_estimators=100)
clf6a.fit(X_train, y_train)
clf6a_predictions = clf6a.predict(X_test)
clf6a_accuracy = accuracy_score(y_test, clf6a_predictions)
clf6a_f1 = f1_score(y_test, clf6a_predictions)
print("AdaBoost Classifier DT: [Accuracy: {:.4f}, f1-score: {:.4f}]".format(clf6a_accuracy, clf6a_f1))

"""AdaBoost Classifier with NaiveBayes base"""
clf6b = AdaBoostClassifier(n_estimators=100, base_estimator=GaussianNB())
clf6b.fit(X_train, y_train)
clf6b_predictions = clf6b.predict(X_test)
clf6b_accuracy = accuracy_score(y_test, clf6b_predictions)
clf6b_f1 = f1_score(y_test, clf6b_predictions)
print("AdaBoost Classifier NB: [Accuracy: {:.4f}, f1-score: {:.4f}]".format(clf6b_accuracy, clf6b_f1))

"""Gradient Boosting Classifier"""
clf7 = GradientBoostingClassifier(n_estimators=100)
clf7.fit(X_train, y_train)
clf7_predictions = clf7.predict(X_test)
clf7_accuracy = accuracy_score(y_test, clf7_predictions)
clf7_f1 = f1_score(y_test, clf7_predictions)
print("Gradient Boosting Classifier: [Accuracy: {:.4f}, f1-score: {:.4f}]".format(clf7_accuracy, clf7_f1))