import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

SP500 = pd.read_csv('SP500_2.csv', parse_dates=True)
Nasdaq = pd.read_csv('Nasdaq_2.csv', parse_dates=True)
DJI = pd.read_csv('DJI_2.csv', parse_dates=True)
DAX = pd.read_csv('DAX_2.csv', parse_dates=True)
Paris = pd.read_csv('Paris_2.csv', parse_dates=True)
Tokyo = pd.read_csv('Tokyo_2.csv', parse_dates=True)
HongKong = pd.read_csv('HongKong_2.csv', parse_dates=True)
Aus = pd.read_csv('Aus_2.csv', parse_dates=True)
megaset = pd.read_csv('megaset_2.csv')

"""Apply PCA by fitting the data with only 10 dimensions"""
pca = PCA(n_components = 10)
pca.fit(megaset)
print(pca.explained_variance_ratio_)
print("Our reduced dimensions can explain {:.4f}".format(sum(pca.explained_variance_ratio_)),
      "% of the variance in the original data")

reduced_megaset = pca.transform(megaset)
reduced_megaset = pd.DataFrame(reduced_megaset, columns = ['Dimension 1', 'Dimension 2',
                                                           'Dimension 3', 'Dimension 4',
                                                           'Dimension 5', 'Dimension 6',
                                                           'Dimension 7', 'Dimension 8',
                                                           'Dimension 9', 'Dimension 10'])


"""Our target variable is tomorrow's Adj Close"""
target_raw = (SP500['Adj Close'].shift(-1)/SP500['Adj Close'])-1

"""Label encode our target variable, 1 for increase, 0 for decrease or no change"""
target = target_raw[Max:]
target[target > 0] = 1
target[target <= 0] = 0

"""Split our dimension-reduced megaset into training and cross-validation (test) subsets"""
X_train = reduced_megaset[:6001]
X_test = reduced_megaset[6001:-1]
y_train = target[:6001]
y_test = target[6001:-1]

"""Visualisation of PCA-transformed dimensions"""
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1],
            c=y_train, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();