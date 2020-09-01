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

import pandas as pd

def compile_data(symbols, period_start, period_end, pwd):
    """
    This function helps user to 

    Parameters
    ----------
    symbols : List
        A list of symbols that represent the stocks of interest.
    period_start : String
        Period start date of the output table.
    period_end : TYPE
        Period end date of the output table.
    pwd : String
        Directory path where the stock price CSV files are stored.

    Returns
    -------
    df : Pandas DataFrame
        A Pandas DataFrame containing stock price information within the
        defined window of interest.

    """
    df = pd.DataFrame()
    
    dates = [x.strftime('%Y-%m-%d') for x in list(pd.date_range(period_start, period_end))]
    df['dates'] = dates
    for i in symbols:
        temp = pd.read_csv(glob('{}{}*.csv'.format(pwd, i))[0], index_col=0)[['date','adjusted close']]
        temp.columns = ['dates', i]
        df = df.merge(temp, on='dates', how='left')
        
    df.columns = ['dates'] + symbols
    return df