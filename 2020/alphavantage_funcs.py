"""
Copyright (c) 2020, Heung Kit Leslie Chung
All rights reserved.
Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
"""

import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import timedelta
from datetime import datetime
import time

def request_stock_price_hist(symbol, token, sample = False):
    """
    This function helps the user to retrieve historical stock prices for the
    specified symbol from Alpha Vantage.

    Parameters
    ----------
    symbol : String
        Stock symbol, e.g. Apple is AAPL.
    token : String
        Register an account on alphavantage.co and get your API.
    sample : Boolean, optional
        Set to True to take a sample of the data only.

    Returns
    -------
    df : Pandas DataFrame
        A Pandas DataFrame containing stock price information.

    """
    if sample == False:
        q_string = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={}&outputsize=full&apikey={}'
    else:
        q_string = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={}&apikey={}'
    
    print("Retrieving stock price data from Alpha Vantage (This may take a while)...")
    r = requests.get(q_string.format(symbol, token))
    print("Data has been successfully downloaded...")
    date = []
    colnames = list(range(0, 7))
    df = pd.DataFrame(columns = colnames)
    print("Sorting the retrieved data into a dataframe...")
    for i in tqdm(r.json()['Time Series (Daily)'].keys()):
        date.append(i)
        row = pd.DataFrame.from_dict(r.json()['Time Series (Daily)'][i], orient='index').reset_index().T[1:]
        df = pd.concat([df, row], ignore_index=True)
    df.columns = ["open", "high", "low", "close", "adjusted close", "volume", "dividend amount", "split cf"]
    df['date'] = date
    return df

def request_quote(symbol, token):
    """
    This function helps the user to retrieve current stock quote for the
    specified symbol from Alpha Vantage.

    Parameters
    ----------
    symbol : String
        Stock symbol, e.g. Apple is AAPL.
    token : String
        Register an account on alphavantage.co and get your API.

    Returns
    -------
    df : Pandas DataFrame
        A Pandas DataFrame containing stock price information.

    """
    q_string = 'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={}&apikey={}'
    r = requests.get(q_string.format(symbol, token))
    colnames = [x.split('. ')[1] for x in list(r.json()['Global Quote'].keys())]
    df = pd.DataFrame.from_dict(r.json()['Global Quote'], orient='index').reset_index().T[1:]
    df.columns = colnames
    return df

def request_symbols(token):
    """
    This function helps the user to retrieve all currently active listed 
    stocks and their symbols from Alpha Vantage.

    Parameters
    ----------
    token : String
        Register an account on alphavantage.co and get your API.

    Returns
    -------
    df : Pandas DataFrame
        A Pandas DataFrame containing stock price information.

    """
    q_string = 'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={}'   
    print("Retrieving stock symbols from Alpha Vantage...")
    r = requests.get(q_string.format(token)).content
    print("Data has been successfully downloaded...")
    r = r.decode("utf-8")
    colnames = list(range(0, 6))
    df = pd.DataFrame(columns = colnames)
    print("Sorting the retrieved data into a dataframe...")
    for i in tqdm(range(1, len(r.split('\r\n'))-1)):
        row = pd.DataFrame(r.split('\r\n')[i].split(',')).T
        df = pd.concat([df, row], ignore_index=True)
    df.columns = r.split('\r\n')[0].split(',')
    return df

#symb = list(sym_df.loc[sym_df['exchange'] == 'NASDAQ']['symbol'])
#save_stock_price_hist(symb[116:200], token, 'Data')

def save_stock_price_hist(symbol_list, token, pwd=''):
    """
    This function helps the user to download all historical stock prices
    for each symbol on the symbol_list parameter from Alpha Vantage. There 
    will be a CSV file for each symbol and the user can specify the output 
    path for the CSV files.
    
    Please note that Alpha Vantage does not accept more than 5 calls per
    minute. Therefore if your symbol_list object is long, the function will
    automatically set waiting time to ensure there is no break once the
    function is set to run.

    Parameters
    ----------
    symbol_list : List
        A list of symbols (string type) to query.  
    token : String
        Register an account on alphavantage.co and get your API.
    pwd : String, optional
        The path to the directory where the CSV files will be saved.

    Returns
    -------
    None

    """
    n = datetime.now()
    curr = 0
    for i in symbol_list:
        ind = symb.index(i)
        l = datetime.now()
        if (l - n < timedelta(minutes=1)) and (ind-curr == 5):
            print("We made 5 API calls in the last minute, taking a break...")
            time.sleep((timedelta(minutes=2) - (l - n)).seconds)
            curr = i
        else:
            df = request_stock_price_hist(i, token)
            df.to_csv('{}/{}_{}.csv'.format(pwd, i, datetime.today().strftime('%Y%m%d')))
    return None
    