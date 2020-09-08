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

def request_news(start_date, end_date, topic, token):
    """
    This function helps the user to extract news articles from a given period
    regarding the topic through Contextual Web Search API. 

    Parameters
    ----------
    start_date : String
        The start date of the period of interest. e.g. 2020-01-01
    end_date : String
        The end date of the period of interest. e.g. 2020-01-31
    topic : String
        The topic of interest. e.g. NASDAQ
    token : String
        Your Contextual Web Search token on RapidAPI.

    Returns
    -------
    big_df : Pandas DataFrame
        A Pandas DataFrame containing news articles within the defined window 
        of interest.

    """
    delt = (datetime.strptime(end_date, '%Y-%m-%d').date() - datetime.strptime(start_date, '%Y-%m-%d').date()).days
    date_list = [(str((datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=x)).date())) for x in range(0, delt, 7)]
    
    big_df = pd.DataFrame()
    
    for j in range(0, len(date_list)-1):
        start_d = date_list[j]
        end_d = date_list[j+1]
    
        url = "https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/Search/NewsSearchAPI"
        querystring = {"fromPublishedDate":start_d,"toPublishedDate":end_d,"autoCorrect":"false","pageNumber":"1","pageSize":"50","q":topic,"safeSearch":"false"}
        headers = {
            'x-rapidapi-host': "contextualwebsearch-websearch-v1.p.rapidapi.com",
            'x-rapidapi-key': token
        }
        r = requests.request("GET", url, headers=headers, params=querystring)
        colnames = list(range(0, 11))
        df = pd.DataFrame(columns = colnames)
        no_news = len(r.json()['value'])
        for i in range(0, no_news):
            row = pd.DataFrame(pd.DataFrame.from_dict(r.json()['value'][i], orient='index').reset_index().T.iloc[1])
            df = pd.concat([df, row.T], ignore_index=True)
        
        df.columns = list(pd.DataFrame.from_dict(r.json()['value'][0], orient='index').reset_index().T.iloc[0])
        df = df[['title', 'url', 'description', 'body', 'datePublished']]
        
        big_df = pd.concat([big_df, df], ignore_index=True)
    
    big_df['dates'] = [x[0:10] for x in big_df['datePublished']]
    del(big_df['datePublished'])
    
    return big_df

