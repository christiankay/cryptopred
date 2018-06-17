# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:58:42 2018

@author: Chris
"""
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np
import data_reader as data_reader

df_ini = data_reader.get_poloniex_data()
coin_list = df_ini.fetch_coin_data(['BTC', 'LTC', 'ETH', 'XMR'])

def create_coin_dataset(coin_list):
    
    
    for coin in coin_list:

        bitcoin_market_info = df["BTC"]
        eth_market_info = df["ETH"]


        #### preparing data
        bitcoin_market_info.columns =[bitcoin_market_info.columns[0]]+['bt_'+i for i in bitcoin_market_info.columns[1:]]
        eth_market_info.columns =[eth_market_info.columns[0]]+['eth_'+i for i in eth_market_info.columns[1:]]
        bitcoin_market_info.rename(columns={'bt_date': 'date'}, inplace=True)
        eth_market_info.rename(columns={'eth_date': 'date'}, inplace=True)
        
        
        market_info = pd.merge(bitcoin_market_info,eth_market_info, on=['date'])
        market_info = market_info[market_info['date']>='2017-01-01']
        for coins in ['bt_', 'eth_']: 
            kwargs = { coins+'day_diff': lambda x: (x[coins+'close']-x[coins+'open'])/x[coins+'open']}
            market_info = market_info.assign(**kwargs)
        market_info.head()
        
        for coins in ['bt_', 'eth_']: 
            kwargs = { coins+'close_off_high': lambda x: 2*(x[coins+'high']- x[coins+'close'])/(x[coins+'high']-x[coins+'low'])-1,
                    coins+'volatility': lambda x: (x[coins+'high']- x[coins+'low'])/(x[coins+'open'])}
            market_info = market_info.assign(**kwargs)
        
        
        
        model_data = market_info[['date']+[coin+metric for coin in ['bt_', 'eth_'] 
                                           for metric in ['close','volume','close_off_high','volatility']]]
        # need to reverse the data frame so that subsequent rows represent later timepoints
        model_data = model_data.sort_values(by='date')
        model_data.head()
        
        # we don't need the date columns anymore
        split_date = '2017-06-01'
        training_set, test_set = model_data[model_data['date']<split_date], model_data[model_data['date']>=split_date]
        training_set = training_set.drop('date', 1)
        test_set = test_set.drop('date', 1)
        
        window_len = 10
        norm_cols = [coin+metric for coin in ['bt_', 'eth_'] for metric in ['close','volume']]


import numpy as np

from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.models import Range1d
from bokeh.embed import components


def datetime(x):
    return np.array(x, dtype=np.datetime64)

p1 = figure(x_axis_type="datetime", title="Stock Closing Prices")
p1.grid.grid_line_alpha=0.3
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Price'

BTC = coin_list['BTC']
LTC = coin_list['LTC']
ETH = coin_list['ETH']
XMR = coin_list['XMR']



file = pd.HDFStore('\\\\192.168.0.24\\SambaPi\\TweetDat\\tweets_BITCOIN.h5')
for key in file:
    data = pd.DataFrame()
    data[key] = pd.read_hdf(file, key)


p1.line(datetime(data.index.values), data[data.columns[0]], color='#A6CEE3', legend='BTC')
p1.line(datetime(LTC['date_format']), LTC['close'], color='#B2DF8A', legend='LTC')
p1.line(datetime(ETH['date_format']), ETH['close'], color='#33A02C', legend='ETH')
p1.line(datetime(XMR['date_format']), XMR['close'], color='#FB9A99', legend='XMR')
p1.legend.location = "top_left"

aapl = np.array(data[data.columns[0]])
aapl_dates = np.array(BTC.index.values, dtype=np.datetime64)

window_size = 30
window = np.ones(window_size)/float(window_size)
aapl_avg = np.convolve(aapl, window, 'same')

p2 = figure(x_axis_type="datetime", title="BTC One-Month Average")
p2.grid.grid_line_alpha = 0
p2.xaxis.axis_label = 'Date'
p2.yaxis.axis_label = 'Price'
p2.ygrid.band_fill_color = "olive"
p2.ygrid.band_fill_alpha = 0.1

p2.circle(aapl_dates, aapl, size=4, legend='close',
          color='darkgrey', alpha=0.2)

p2.line(aapl_dates, aapl_avg, legend='avg', color='navy')
p2.legend.location = "top_left"

output_file("stocks.html", title="stocks.py example")

show(gridplot([[p1,p2]], plot_width=900, plot_height=900))  # open a browser