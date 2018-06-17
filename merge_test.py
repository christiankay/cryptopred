# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 15:28:18 2018

@author: Chris
"""

import data_reader as data_reader
import Load_Tweets_Class as LTC

import pandas as pd

selected_coins = ['BTC', 'ETH']   #['BTC', 'LTC', 'ETH', 'XMR']
period = 7200#86400

split_date = '2018-06-15' 
        
df_ini = data_reader.get_poloniex_data()
coin_list = df_ini.fetch_coin_data(selected_coins, period)  




       ### init 
get_tweet_data = LTC()
### fetch data based on query word
#get_tweet_data.fetch_tweets(query='BITCOIN', count=100, pages=1)


#get_tweet_data.fetch_stocktwits(query='BTC.X')
### delete duplicates data from CSV
data = get_tweet_data.read_and_clean_data_from_csv(query='BITCOIN')
#data = get_tweet_data.data
        

results = get_tweet_data.analyze_Tweets(data)


merge=pd.merge(df,d, how='inner', on='date')