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
coin_list = df_ini.fetch_coin_data(['BTC', 'LTC', 'ETH', 'XMR'], period = 86400)

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


split_date = '2017-06-01 00:00:00'



p1 = figure(x_axis_type="datetime", title="Stock Closing Prices")
p1.grid.grid_line_alpha=0.3
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Price'

BTC = coin_list['BTC']
LTC = coin_list['LTC']
ETH = coin_list['ETH']
XMR = coin_list['XMR']

data = pd.merge(left=BTC, left_on='date_format',
         right=ETH, right_on='date_format')
data = data.dropna()
data = data[data['date_format']>split_date]
t = pd.to_datetime(data['date_format'])

from matplotlib.dates import DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates



 
def pandas_candlestick_ohlc(dat, stick = "day", otherseries = None):
    """
    :param dat: pandas DataFrame object with datetime64 index, and float columns "Open", "High", "Low", and "Close", likely created via DataReader from "yahoo"
    :param stick: A string or number indicating the period of time covered by a single candlestick. Valid string inputs include "day", "week", "month", and "year", ("day" default), and any numeric input indicates the number of trading days included in a period
    :param otherseries: An iterable that will be coerced into a list, containing the columns of dat that hold other series to be plotted as lines
 
    This will show a Japanese candlestick plot for stock data stored in dat, also plotting other series if passed.
    """
    mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
    alldays = DayLocator()              # minor ticks on the days
    dayFormatter = DateFormatter('%d')      # e.g., 12
 
    # Create a new DataFrame which includes OHLC data for each period specified by stick input
    transdat = dat.loc[:,["open", "high", "low", "close"]]
    print("transdat",transdat.head() )
    if (type(stick) == str):
        if stick == "day":
            plotdat = transdat
            stick = 1 # Used for plotting
        elif stick in ["week", "month", "year"]:
            if stick == "week":
                transdat["week"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[1]) # Identify weeks
            elif stick == "month":
                transdat["month"] = pd.to_datetime(transdat.index).map(lambda x: x.month) # Identify months
            transdat["year"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[0]) # Identify years
            grouped = transdat.groupby(list(set(["year",stick]))) # Group by year and other appropriate variable
            plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) # Create empty data frame containing what will be plotted
            for name, group in grouped:
                plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0,0],
                                            "High": max(group.High),
                                            "Low": min(group.Low),
                                            "Close": group.iloc[-1,3]},
                                           index = [group.index[0]]))
            if stick == "week": stick = 5
            elif stick == "month": stick = 30
            elif stick == "year": stick = 365
 
    elif (type(stick) == int and stick >= 1):
        transdat["stick"] = [np.floor(i / stick) for i in range(len(transdat.index))]
        grouped = transdat.groupby("stick")
        plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) # Create empty data frame containing what will be plotted
        for name, group in grouped:
            plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0,0],
                                        "High": max(group.High),
                                        "Low": min(group.Low),
                                        "Close": group.iloc[-1,3]},
                                       index = [group.index[0]]))
 
    else:
        raise ValueError('Valid inputs to argument "stick" include the strings "day", "week", "month", "year", or a positive integer')
 
 
    # Set plot parameters, including the axis object ax used for plotting
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)

    if plotdat.index[-1] - plotdat.index[0] < pd.Timedelta('730 days'):
        weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_minor_locator(alldays)
    else:
        weekFormatter = DateFormatter('%b %d, %Y')
    ax.xaxis.set_major_formatter(weekFormatter)
 
    ax.grid(True)
 
    # Create the candelstick chart
    candlestick_ohlc(ax, list(zip(list(mdates.date2num(plotdat.index.tolist())), plotdat["open"].tolist(), plotdat["high"].tolist(),
                      plotdat["low"].tolist(), plotdat["close"].tolist())),
                      colorup = "black", colordown = "red", width = stick * .4)
 
    # Plot other series (such as moving averages) as lines
    if otherseries != None:
        if type(otherseries) != list:
            otherseries = [otherseries]
        dat.loc[:,otherseries].plot(ax = ax, lw = 1.3, grid = True)
 
    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
 
    plt.show()
    
    
#BTC = pd.to_datetime(BTC['date_format'])    
c = BTC.set_index(datetime(BTC['date_format'])) 
pandas_candlestick_ohlc(c)