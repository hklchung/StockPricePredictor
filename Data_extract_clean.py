from scipy.stats import skew
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def get_data():
    """First get our csv files and import as DataFrames"""
    SP500=pd.read_csv('SP500.csv', index_col='Date', parse_dates=True)
    Nasdaq=pd.read_csv('NASDAQ.csv', index_col='Date', parse_dates=True)
    DJI=pd.read_csv('DJI.csv', index_col='Date', parse_dates=True)
    DAX=pd.read_csv('DAX.csv', index_col='Date', parse_dates=True)
    Paris=pd.read_csv('CAC40.csv', index_col='Date', parse_dates=True)
    Tokyo=pd.read_csv('N225.csv', index_col='Date', parse_dates=True)
    HongKong=pd.read_csv('HSI.csv', index_col='Date', parse_dates=True)
    Aus=pd.read_csv('ASX.csv', index_col='Date', parse_dates=True)
    """Get rid of the first year because ASX has no data"""
    SP500 = SP500[502:]
    """Fill in missing data by forward fill"""
    SP500.fillna(method='ffill',inplace=True)
    Nasdaq.fillna(method='ffill',inplace=True)
    DJI.fillna(method='ffill',inplace=True)
    DAX.fillna(method='ffill',inplace=True)
    Paris.fillna(method='ffill',inplace=True)
    Tokyo.fillna(method='ffill',inplace=True)
    HongKong.fillna(method='ffill',inplace=True)
    Aus.fillna(method='ffill',inplace=True)
    return SP500, Nasdaq, DJI, DAX, Paris, Tokyo, HongKong, Aus

def left_join(mother, child):
    """This function grabs data from all dfs on days SP500 was traded"""
    df_temp = pd.DataFrame(index = mother.index)
    df_temp1 = df_temp.join(child)
    df_temp1 = df_temp1.replace('null', np.nan)
    df_temp1.fillna(method='ffill', inplace=True)
    df_temp1.fillna(method='backfill', inplace=True)
    return df_temp1

Nasdaq_new = left_join(SP500, Nasdaq)
DJI_new = left_join(SP500, DJI)
DAX_new = left_join(SP500, DAX)
Paris_new = left_join(SP500, Paris)
Tokyo_new = left_join(SP500, Tokyo)
HongKong_new = left_join(SP500, HongKong)
Aus_new = left_join(SP500, Aus)


def reset_index(df):
    """Dates are no longer important"""
    df['Date'] = df.index
    df = df.reset_index(level=['Date'])
    return df
    
SP500 = reset_index(SP500)
Nasdaq = reset_index(Nasdaq_new)
DJI = reset_index(DJI_new)
DAX = reset_index(DAX_new)
Paris = reset_index(Paris_new)
Tokyo = reset_index(Tokyo_new)
HongKong = reset_index(HongKong_new)
Aus = reset_index(Aus_new)


SP500.to_csv('SP500_new.csv', index=False)
Nasdaq.to_csv('Nasdaq_new.csv', index=False)
DJI.to_csv('DJI_new.csv', index=False)
DAX.to_csv('DAX_new.csv', index=False)
Paris.to_csv('Paris_new.csv', index=False)
Tokyo.to_csv('Tokyo_new.csv', index=False)
HongKong.to_csv('HongKong_new.csv', index=False)
Aus.to_csv('Aus_new.csv', index=False)