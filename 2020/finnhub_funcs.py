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

symbol = 'AAPL'
resolution = 1

def request_quote(symbol, resolution, token):
    """
    Enter stock symbol, resolution and your finnhub token to retrieve
    stock price.

    Parameters
    ----------
    symbol : String
        Stock symbol, e.g. Apple is AAPL.
    resolution : Integer or String
        The possible values are 1, 5, 15, 30, 60, D, W, M.
    token : String
        Register an account on finnhub.io and get your API.

    Returns
    -------
    df : Pandas DataFrame
        A Pandas DataFrame containing stock price information.

    """
    q_string = 'https://finnhub.io/api/v1/stock/candle?symbol={}&resolution={}&from=1572651390&to=1572910590&token={}'
    r = requests.get(q_string.format(symbol, resolution, token))
    c = []
    [c.append(x) for x in r.json().keys()]
    df = pd.DataFrame.from_dict(r.json(), orient='index').reset_index().T[1:]
    df.columns = c
    del(df['s'])
    return df

def request_comp_info(symbol, token):
    """
    Enter stock symbol and your finnhub token to retrieve information about
    the company.

    Parameters
    ----------
    symbol : String
        Stock symbol, e.g. Apple is AAPL.
    token : String
        Register an account on finnhub.io and get your API.

    Returns
    -------
    df : Pandas DataFrame
        A Pandas DataFrame containing information about a company.

    """
    q_string = 'https://finnhub.io/api/v1/stock/profile2?symbol={}&token={}'
    r = requests.get(q_string.format(symbol, token))
    c = []
    [c.append(x) for x in r.json().keys()]
    df = pd.DataFrame.from_dict(r.json(), orient='index').reset_index().T[1:]
    df.columns = c
    return df