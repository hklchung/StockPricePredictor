from scipy.stats import skew
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""First get our csv files and import as DataFrames"""
SP500 = pd.read_csv('SP500_new.csv', parse_dates=True)
Nasdaq = pd.read_csv('Nasdaq_new.csv', parse_dates=True)
DJI = pd.read_csv('DJI_new.csv', parse_dates=True)
DAX = pd.read_csv('DAX_new.csv', parse_dates=True)
Paris = pd.read_csv('Paris_new.csv', parse_dates=True)
Tokyo = pd.read_csv('Tokyo_new.csv', parse_dates=True)
HongKong = pd.read_csv('HongKong_new.csv', parse_dates=True)
Aus = pd.read_csv('Aus_new.csv', parse_dates=True)

"""Our target variable is tomorrow's Adj Close"""
target_raw = (SP500['Adj Close'].shift(-1)/SP500['Adj Close'])-1

datasets = [SP500, Nasdaq, DJI, DAX, Paris, Tokyo, HongKong, Aus]
names = ['SP500', 'Nasdaq', 'DJI', 'DAX', 'Paris', 'Tokyo', 'HongKong', 'Aus']

"""
def compute_daily_returns(dataset):
    dataset['daily_returns'] = (dataset['Adj Close']/dataset['Adj Close'].shift(1))-1
    dataset = dataset.set_value('1990-12-31', 'daily_returns', 0)
    return dataset
"""

"""
The generate_features function performs feature engineering using Adj Close,
the features generated are Daily Returns, Momentum (Daily Returns over 2 days),
Daily Return SMA and lagging Daily Returns
"""

"""
Because previously we had some "null" values, the DataFrames columns are still considered strings,
we need to change that so we change the values to float
"""


def generate_features(datasets, DR, DR_SMA, Lagging):
    Max = max(DR, DR_SMA, Lagging+1)
    for i in range(len(datasets)):
        dataset = datasets[i]
        name = names[i]
        for j in range(1, DR+1):
            dataset[name+'_'+str(j)+'DailyReturn'] = (dataset['Adj Close'].astype(float)/dataset['Adj Close'].astype(float).shift(j))-1
        for k in range(2, DR_SMA+1):
            dataset[name+'_'+str(k)+'DR_SMA'] = pd.rolling_mean(dataset[name+'_'+str(1)+'DailyReturn'], window=k)
        for l in range(1, Lagging+1):
            dataset[name+'_'+str(l)+'LaggingDays'] = dataset[name+'_'+str(1)+'DailyReturn'].shift(l)
        dataset.drop(dataset.index[:Max], inplace=True)
    return Max

"""After feature engineering, merge all datasets and drop the 'useless' features"""
def merge_datasets(datasets):
    drop_features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Date']
    for i in range(len(datasets)):
        datasets[i] = datasets[i].drop(drop_features, axis=1)
    megaset = pd.concat(datasets, axis=1)
    return megaset

generate_features(datasets, 20, 20, 20)
megaset = merge_datasets(datasets)

"""Label encode our target variable, 1 for increase, 0 for decrease or no change"""
target = target_raw[Max:] #use Max: if done before generate_features()
target[target > 0] = 1
target[target <= 0] = 0

"""Split our megaset into training and cross-validation (test) subsets"""
X_train = reduced_megaset[:6001]
X_test = reduced_megaset[6001:-1]
y_train = target[:6001]
y_test = target[6001:-1]
