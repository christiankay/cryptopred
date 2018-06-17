# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:23:29 2018

@author: Chris
"""

import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np


class get_cmc_data:
    
    def __init__(self, startdate = 20130428, enddate=time.strftime("%Y%m%d")):
       
        
        self.startdate = startdate
        self.enddate = enddate
        
        
    def read_coin_data(self, coin="bitcoin"):
        print("Start reading coin data...")
        print("start date: ", self.startdate)
        print("end date: ", time.strftime("%Y%m%d"))
        # get market info for bitcoin from the start of 2016 to the current day
        coin_market_info = pd.read_html("https://coinmarketcap.com/currencies/"+coin+"/historical-data/?start="+str(self.startdate)+"&end="+self.enddate)[0]
        # convert the date string to the correct date format
        coin_market_info = coin_market_info.assign(Date=pd.to_datetime(coin_market_info['Date']))
        # when Volume is equal to '-' convert it to 0
#        try:
#            coin_market_info.loc[coin_market_info['Volume']=="-",'Volume']=0
#                    # convert to int
#            coin_market_info['Volume'] = coin_market_info['Volume'].astype('int64')
#            # look at the first few rows
#
#        except:
#            print("TypeError: invalid type comparison")

        
        print ("Successfully loaded data: ")
        print (coin+" first few rows: ")
        print (coin_market_info.head())
        
#        # get market info for ethereum from the start of 2016 to the current day
#        eth_market_info = pd.read_html("https://coinmarketcap.com/currencies/ethereum/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]
#        # convert the date string to the correct date format
#        eth_market_info = eth_market_info.assign(Date=pd.to_datetime(eth_market_info['Date']))
#        # look at the first few rows
#        eth_market_info.head()
#        
#        print ("Successfully loded data: ")
#        print ("ethereum first few rows: ")
#        print(eth_market_info.head())
        
        return coin_market_info
        
dfclass = get_cmc_data(startdate = 20130428, enddate=time.strftime("%Y%m%d"))

df_btc = dfclass.read_coin_data("bitcoin")
df_eth = dfclass.read_coin_data("ethereum")