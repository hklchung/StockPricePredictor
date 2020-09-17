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
from glob import glob
from datetime import timedelta
from datetime import datetime
from alphavantage_funcs import *
from news_funcs import *
from compile_dataset import *

# First define a list of stocks we want
symb_list = ['FB', 'NFLX', 'GOOGL', 'MSFT', 'AMZN', 'ADBE']

# Extract and download stock price history to Data/ folder
save_stock_price_hist(symb_list, alphavantage_token, pwd='Data')

# Put together a dataframe with all stock prices
stocks = compile_data(symb_list, '2020-01-01', '2020-05-31', 'Data/')

# Extract news about NASDAQ
news = request_news('2020-01-01', '2020-5-31', 'NASDAQ', rapidapi_contextualweb_token)
news['positivity'] = [sentiment(x)[0] for x in news['body']]
news['subjectivity'] = [sentiment(x)[1] for x in news['body']]
news['tally'] = 1

# Aggregate news at daily level
agg_news = pd.DataFrame(news.groupby('dates').agg(avg_pos=pd.NamedAgg(column='positivity', aggfunc=np.mean),
                                       avg_subj=pd.NamedAgg(column='subjectivity', aggfunc=np.mean),
                                       news_cnt=pd.NamedAgg(column='tally', aggfunc=sum))).reset_index(level=0)

# Join news to stock price history data
stocks = stocks.merge(agg_news[['dates', 'avg_pos', 'avg_subj', 'news_cnt']], on='dates', how='left')