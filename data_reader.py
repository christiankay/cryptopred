# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:42:54 2018

@author: Chris
"""

import json
import numpy as np
import os
import pandas as pd
import urllib.request
import datetime
import time

class get_poloniex_data:
    
    def __init__(self, startdate = 20130428, enddate=time.strftime("%Y%m%d")):
       
      
        self.startdate = startdate
        self.enddate = enddate
        #Path to store cached currency data
        self.datPath = 'CurDat/'
        if not os.path.exists(self.datPath):
            os.mkdir(self.datPath)

    def fetch_coin_data(self, coins=['BTC', 'LTC', 'ETH', 'XMR'], period = 7200):
        print("start date: ", self.startdate)
        print("end date: ", time.strftime("%Y%m%d"))
        print("Start fetching coin data...")
        D = {}
        self.period = period 
        for coin in coins:
    
            dfp = os.path.join(self.datPath, coin + str(self.period) + '.csv')
            try:
                df = pd.read_csv(dfp, sep = ',')
            except FileNotFoundError:
                df = self.GetCurDF(coin, dfp)
            D[coin] = df
            print ("Successfully loaded data: ", coin)

#        #Only keep range of data that is common to all currency types
#        cr = min(Di.shape[0] for Di in D)
#        for i in range(len(coins)):
#            D[i] = D[i][(D[i].shape[0] - cr):]
                
        return D    
            
 
    def JSONDictToDF(self, d):
        '''
        Converts a dictionary created from json.loads to a pandas dataframe
        d:      The dictionary
        '''
        n = len(d)
        cols = []
        if n > 0:   #Place the column in sorted order
            cols = sorted(list(d[0].keys()))
        df = pd.DataFrame(columns = cols, index = range(n))
        for i in range(n):
            for coli in cols:
                df.set_value(i, coli, d[i][coli])
        return df
         
    def GetAPIUrl(self, cur):
        '''
        Makes a URL for querying historical prices of a cyrpto from Poloniex
        cur:    3 letter abbreviation for cryptocurrency (BTC, LTC, etc)
     
        returnChartData
        Returns candlestick chart data. Required GET parameters are "currencyPair", "period" (candlestick period in seconds; valid values are 300, 900, 1800, 7200, 14400, and 86400), "start", and "end". "Start" and "end" are given in UNIX timestamp format and used to specify the date range for the data returned. Sample output:
        
        [{"date":1405699200,"high":0.0045388,"low":0.00403001,"open":0.00404545,"close":0.00427592,"volume":44.11655644,
        "quoteVolume":10259.29079097,"weightedAverage":0.00430015}, ...]
        
        Call: https://poloniex.com/public?command=returnChartData&currencyPair=BTC_XMR&start=1405699200&end=9999999999&period=14400
        '''
      
        u = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_' + cur + '&start=1420070400&end=9999999999&period='+str(self.period)
        return u
     
    def GetCurDF(self, cur, fp):
        '''
        cur:    3 letter abbreviation for cryptocurrency (BTC, LTC, etc)
        fp:     File path (to save price data to CSV)
        '''
        openUrl = urllib.request.urlopen(self.GetAPIUrl(cur))
        r = openUrl.read()
        openUrl.close()
        d = json.loads(r.decode())
        df = self.JSONDictToDF(d)
                # convert the date string to the correct date format
        df['date_format'] = pd.to_datetime(df['date'],unit='s')
        df.to_csv(fp, sep = ',')
        return df
     

class PastSampler:
    '''
    Forms training samples for predicting future values from past value
    '''
     
    def __init__(self, N, K):
        '''
        Predict K future sample using N previous samples
        '''
        self.K = K
        self.N = N
 
    def transform(self, A, Y = None):
        M = self.N + self.K     #Number of samples per row (sample + target)
        #Matrix of sample indices like: {{1, 2..., M}, {2, 3, ..., M + 1}}
        I = np.arange(M) + np.arange(A.shape[0] - M + 1).reshape(-1, 1)
        B = A[I].reshape(-1, M * A.shape[1], *A.shape[2:])
        ci = self.N * A.shape[1]    #Number of features per sample
        return B[:, :ci], B[:, ci:] #Sample matrix, Target matrix    
    



if __name__ == "__main__":    
    df_ini = get_poloniex_data()    
    D =  df_ini.fetch_coin_data(['BTC', 'LTC', 'ETH', 'XMR'])
    #Different cryptocurrency types
    cl = ['BTC', 'LTC', 'ETH', 'XMR']
    #Columns of price data to use
    CN = ['close', 'high', 'low', 'open', 'volume']
    #Features are channels
    C = np.hstack((Di[CN] for Di in D))[:, None, :]
    HP = 20                #Holdout period
    A = C[0:-HP]                
    SV = A.mean(axis = 0)   #Scale vector
    C /= SV                 #Basic scaling of data
    #Make samples of temporal sequences of pricing data (channel)
    NPS, NFS = 256, 16         #Number of past and future samples
    ps = PastSampler(NPS, NFS)
    B, Y = ps.transform(A)
    
    
    
    #Architecture of the neural network
    from TFANN import ANNR
     
    NC = B.shape[2]
    #2 1-D conv layers with relu followed by 1-d conv output layer
    ns = [('C1d', [8, NC, NC * 2], 4), ('AF', 'relu'), 
          ('C1d', [8, NC * 2, NC * 2], 2), ('AF', 'relu'), 
          ('C1d', [8, NC * 2, NC], 2)]
    #Create the neural network in TensorFlow
    cnnr = ANNR(B[0].shape, ns, batchSize = 32, learnRate = 2e-5, 
                maxIter = 64, reg = 1e-5, tol = 1e-2, verbose = True)
    cnnr.fit(B, Y)  
    
    
    PTS = []                        #Predicted time sequences
    P, YH = B[[-1]], Y[[-1]]        #Most recent time sequence
    for i in range(HP // NFS):  #Repeat prediction
        P = np.concatenate([P[:, NFS:], YH], axis = 1)
        YH = cnnr.predict(P)
        PTS.append(YH)
    PTS = np.hstack(PTS).transpose((1, 0, 2))
    A = np.vstack([A, PTS]) #Combine predictions with original data
    A = np.squeeze(A) * SV  #Remove unittime dimension and rescale
    C = np.squeeze(C) * SV  
    
    import matplotlib.pyplot as mpl
    
    CI = list(range(C.shape[0]))
    AI = list(range(C.shape[0] + PTS.shape[0] - HP))
    NDP = PTS.shape[0] #Number of days predicted
    for i, cli in enumerate(cl):
        fig, ax = mpl.subplots(figsize = (16 / 1.5, 10 / 1.5))
        hind = i * len(CN) + CN.index('high')
        ax.plot(CI[-4 * HP:], C[-4 * HP:, hind], label = 'Actual')
        ax.plot(AI[-(NDP + 1):], A[-(NDP + 1):, hind], '--', label = 'Prediction')
        ax.legend(loc = 'upper left')
        ax.set_title(cli + ' (High)')
        ax.set_ylabel('USD')
        ax.set_xlabel('Time')
        ax.axes.xaxis.set_ticklabels([])
        mpl.show()