import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

"""Benchmark model"""
SP500 = pd.read_csv('SP500_2.csv', parse_dates=True)
target_raw2 = (SP500['Adj Close'].shift(-1)/SP500['Adj Close'])-1
target2 = target_raw2[6001:-1]
target2[target2 > 0] = 1
target2[target2 <= 0] = 0

naive_prediction = pd.rolling_mean(SP500['Adj Close'], window=10)
naive_prediction[naive_prediction > 0] = 1
naive_prediction[naive_prediction <= 0] = 0
naive_prediction = naive_prediction[6001:-1]

"""
Since we are only making predictions for the final year from 19-10-2016 to 19-10-2017
to measure the accuracy and f1-score of our model, we shall also do the same to 
measure the metrics of our benchmark model
"""
print("There are {}".format(target2.shape[0]), "targets.")
print("There are {}".format(naive_prediction.shape[0]), "predictions from the benchmark model.")

bm_accuracy = accuracy_score(target2, naive_prediction)
bm_f1 = f1_score(target2, naive_prediction)
print("Benchmark Model: [Accuracy: {:.4f}, F1-score: {:.4f}]".format(bm_accuracy, bm_f1))