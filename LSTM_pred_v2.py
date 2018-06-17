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
from sklearn.preprocessing import  MaxAbsScaler
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from keras.models import load_model
import os
import tensorflow as tf
import re


def create_coin_dataset(coin_list, starttime = '2018-06-05', split_date = '2018-06-05' , features = ['_close','_volume','_close_off_high','_volatility', '_date']):
    
    market_info_prev = False
    for coin in coin_list:
        
        coin_data = coin_list[coin]
       
        coin_market_info = coin_data.copy()
        #### preparing data
        coin_market_info.columns =[coin_market_info.columns[0]]+[coin+'_'+i for i in coin_market_info.columns[1:]]
        
        kwargs = { coin+'_day_diff': lambda x: (x[coin+'_close']-x[coin+'_open'])/x[coin+'_open']}
        coin_market_info = coin_market_info.assign(**kwargs)
        
    
        kwargs = { coin+'_close_off_high': lambda x: 2*(x[coin+'_high']- x[coin+'_close'])/(x[coin+'_high']-x[coin+'_low'])-1,
        coin+'_volatility': lambda x: (x[coin+'_high']- x[coin+'_low'])/(x[coin+'_open'])}
        coin_market_info = coin_market_info.assign(**kwargs)
     
        kwargs = { coin+'_date': lambda x: (x[coin+'_date']- x[coin+'_date'][0])}
        coin_market_info = coin_market_info.assign(**kwargs)
        

            
        coin_market_info.rename(columns={coin+'_date_format': 'date_format'}, inplace=True)
        if market_info_prev is not False:
            market_info = pd.merge(market_info_prev,coin_market_info, on=['date_format'])
            market_info_prev = market_info
        else:
            market_info_prev = coin_market_info
            
            
   
        

    
    # need to reverse the data frame so that subsequent rows represent later timepoints
    market_info = market_info[market_info['date_format']>=starttime]
    model_data = market_info[['date_format']+[coin+metric for coin in coin_list
                                   for metric in features]]
    for coin in coin_list:
        if coin+'_date' == 'BTC_date':
            dates = np.asarray(model_data[coin+'_date'])
            dates = dates.reshape(-1, 1)
            norm_dates =  MaxAbsScaler().fit_transform(dates)
            
            model_data[coin + '_date'] = norm_dates
        else:
            model_data = model_data.drop(coin + '_date', 1)
    model_data = model_data.sort_values(by='date_format')
    model_data.head()
    
    
    # we don't need the date columns anymore

    training_set, test_set = model_data[model_data['date_format']<split_date], model_data[model_data['date_format']>=split_date]
    
    training_set = training_set.drop('date_format', 1)
    test_set = test_set.drop('date_format', 1)
    
    print ('Selected following features: ', training_set.columns.values.tolist()) 
    print ('Size of training data set: ',  training_set.shape)
    print ('Size of test data set: ',  test_set.shape)
    return training_set, test_set, model_data     




selected_coins = ['BTC', 'ETH']   #['BTC', 'LTC', 'ETH', 'XMR']
period = 7200#86400

split_date = '2018-06-15' 
        
df_ini = data_reader.get_poloniex_data()
coin_list = df_ini.fetch_coin_data(selected_coins, period)   

import raspi_tweet_fetcher.Load_Tweets_Class as LTC    
      
       ### init 
get_tweet_data = LTC.get_tweets()
### fetch data based on query word
#get_tweet_data.fetch_tweets(query='BITCOIN', count=100, pages=1)


#get_tweet_data.fetch_stocktwits(query='BITCOIN')
### delete duplicates data from CSV
data = get_tweet_data.read_and_clean_data_from_csv(query='BITCOIN')
#data = get_tweet_data.data
        

results = get_tweet_data.analyze_Tweets(data)
    
training_set, test_set, model_data  = create_coin_dataset(coin_list, starttime = '2018-06-05',
                                              split_date = split_date , 
                                              features = ['_close','_volume','_volatility','_date'])# '_date','_close_off_high'])        
        
window_len = 15
norm_cols = [coin+metric for coin in selected_coins for metric in ['_close','_volume','_close_off_high','_volatility', '_date']]


## getting the Bitcoin and Eth logos
#import sys
#from PIL import Image
#import io
#
#if sys.version_info[0] < 3:
#    import urllib2 as urllib
#    bt_img = urllib.urlopen("http://logok.org/wp-content/uploads/2016/10/Bitcoin-Logo-640x480.png")
#    eth_img = urllib.urlopen("https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Ethereum_logo_2014.svg/256px-Ethereum_logo_2014.svg.png")
#else:
#    import urllib
#    bt_img = urllib.request.urlopen("http://logok.org/wp-content/uploads/2016/10/Bitcoin-Logo-640x480.png")
#    eth_img = urllib.request.urlopen("https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Ethereum_logo_2014.svg/256px-Ethereum_logo_2014.svg.png")
#
#image_file = io.BytesIO(bt_img.read())
#bitcoin_im = Image.open(image_file)
#
#image_file = io.BytesIO(eth_img.read())
#eth_im = Image.open(image_file)
#width_eth_im , height_eth_im  = eth_im.size
#eth_im = eth_im.resize((int(eth_im.size[0]*0.8), int(eth_im.size[1]*0.8)), Image.ANTIALIAS)
#
#

#
#fig, (ax1, ax2) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]})
#ax1.set_ylabel('Closing Price ($)',fontsize=12)
#ax2.set_ylabel('Volume ($ bn)',fontsize=12)
#ax2.set_yticks([int('%d000000000'%i) for i in range(10)])
#ax2.set_yticklabels(range(10))
#ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
#ax1.set_xticklabels('')
#ax2.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
#ax2.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,7]])
#ax1.plot(bitcoin_market_info['bt_date'].astype(datetime.datetime),bitcoin_market_info['bt_open'])
#ax2.bar(bitcoin_market_info['bt_date'].astype(datetime.datetime).values, bitcoin_market_info['bt_volume'].values)
#fig.tight_layout()
#fig.figimage(bitcoin_im, 100, 120, zorder=3,alpha=.5)
#plt.show()
#
#
#
#fig, (ax1, ax2) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]})
##ax1.set_yscale('log')
#ax1.set_ylabel('Closing Price ($)',fontsize=12)
#ax2.set_ylabel('Volume ($ bn)',fontsize=12)
#ax2.set_yticks([int('%d000000000'%i) for i in range(10)])
#ax2.set_yticklabels(range(10))
#ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
#ax1.set_xticklabels('')
#ax2.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
#ax2.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,7]])
#ax1.plot(eth_market_info['eth_date'].astype(datetime.datetime),eth_market_info['eth_open'])
#ax2.bar(eth_market_info['eth_date'].astype(datetime.datetime).values, eth_market_info['eth_volume'].values)
#fig.tight_layout()
#fig.figimage(eth_im, 300, 180, zorder=3, alpha=.6)
#plt.show()
#
#






LSTM_training_inputs = []
for i in range(len(training_set)-window_len):
    temp_set = training_set[i:(i+window_len)].copy()
#    for col in norm_cols:
#        temp_set = MaxAbsScaler().fit_transform(temp_set)
#        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1 ## sclaling
    LSTM_training_inputs.append(temp_set)




LSTM_test_inputs = []
for i in range(len(test_set)-window_len):
    temp_set = test_set[i:(i+window_len)].copy()
#    for col in norm_cols:
        
#        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1 ##scaling
    
    LSTM_test_inputs.append(temp_set)
    
    



#print("example of input data", LSTM_training_inputs[0])

# I find it easier to work with numpy arrays rather than pandas dataframes
# especially as we now only have numerical data
LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)

LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)

## scaling
for i in range(LSTM_training_inputs.shape[2]):
    MaxAbsScaler().fit_transform(LSTM_training_inputs[:,:,i])
#LSTM_training_inputs = MaxAbsScaler().transform(LSTM_training_inputs)
#LSTM_test_inputs = MaxAbsScaler().transform(LSTM_training_inputs)



# model output is next price normalised to 10th previous closing price
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()    
test_set_close = test_set['ETH_close'].values  
pred = [] 
quo = []
for i in range(len(test_set_close)-window_len):
      
    current= test_set_close[i]
    nxt = test_set_close[i+1]
    diff = nxt - current
    quo.append(diff/current)
    
    if diff/current > 0.01:
        pred.append(2)
    elif diff/current  < -0.01:
        pred.append(1) 
    else:
        pred.append(0)
        
LSTM_test_outputs = pred
LSTM_test_outputs = np.asarray(LSTM_test_outputs)
LSTM_test_outputs_onehot = enc.fit_transform(LSTM_test_outputs.reshape(-1,1)).toarray()

training_set_close = training_set['ETH_close'].values  
pred = [] 
quo = []
for i in range(len(training_set_close)-window_len):
      
    current= training_set_close[i]
    nxt = training_set_close[i+1]
    diff = nxt - current
    quo.append(diff/current)
    
    if diff/current > 0.01:
        pred.append(2)
    elif diff/current  < -0.01:
        pred.append(1) 
    else:
        pred.append(0)
        
LSTM_training_outputs = pred
LSTM_training_outputs = np.asarray(LSTM_training_outputs)
LSTM_training_outputs_onehot = enc.fit_transform(LSTM_training_outputs.reshape(-1,1)).toarray()

class1 = num_ones = (LSTM_training_outputs_onehot[:,0] == 1).sum()
print("data in class1: ", class1)
class2 = num_ones = (LSTM_training_outputs_onehot[:,1] == 1).sum()
print("data in class1: ", class2)
class3 = num_ones = (LSTM_training_outputs_onehot[:,2] == 1).sum()
print("data in class1: ", class3)

datPath = 'models/'
path = os.path.join(datPath, 'ETH_' + str(period) + '_' + str(window_len) + '.h5')

pred_coin = 'ETH_'
try:

    model = load_model(path)
    eth_model = model

    print("Model succesfully loaded")
except:
    print("Could not find saved model data!!")
    print("Start training model!")
    # import the relevant Keras modules
    from keras.models import Sequential
    from keras.layers import Activation, Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    
    def build_model(inputs, output_size, neurons, activ_func="softmax",
                    dropout=0.5, loss='mae', optimizer="adam"):
        model = Sequential()
    
        model.add(LSTM(output_dim=output_size, activation='softmax',input_shape=(inputs.shape[1], inputs.shape[2]),return_sequences=False))
        
        model.add(Dropout(dropout))
        model.add(Dense(units=output_size))
        model.add(Activation(activ_func))
    
        model.compile(loss=loss, optimizer=optimizer,metrics=['accuracy']) #metrics=['accuracy']
        return model
        # random seed for reproducibility
    np.random.seed(252)
    # initialise model architecture
    eth_model = build_model(LSTM_training_inputs, output_size=3, neurons = 200)
    # train model on data
    # note: eth_history contains information on the training error per epoch
    eth_history = eth_model.fit(LSTM_training_inputs, LSTM_training_outputs_onehot, 
                                epochs=2, batch_size=1, verbose=1, shuffle=True,  validation_data=(LSTM_test_inputs, LSTM_test_outputs_onehot))


    if not os.path.exists(datPath):
        os.mkdir(datPath)
    eth_model.save(path)

    print ('model successfully saved!')


    fig, ax1 = plt.subplots(1,1)
    
    ax1.plot(eth_history.epoch, eth_history.history['loss'])
    ax1.plot(eth_history.history['val_loss'])
    ax1.set_title('Training & Test Error')
    
    
    if eth_model.loss == 'mae':
        ax1.set_ylabel('Mean Absolute Error (MAE)',fontsize=12)
    # just in case you decided to change the model loss calculation
    else:
        ax1.set_ylabel('Model Loss',fontsize=12)
    ax1.set_xlabel('# Epochs',fontsize=12)
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig(path+'.png')
    plt.show()


from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

fig, ax1 = plt.subplots(1,1)
ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,5,9]])
ax1.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,5,9]])
ax1.plot(model_data[model_data['date_format']< split_date]['date_format'][window_len:].astype(datetime.datetime),
         training_set[pred_coin+'close'][window_len:], label='Actual')
ax1.plot(model_data[model_data['date_format']< split_date]['date_format'][window_len:].astype(datetime.datetime),
         ((np.transpose(eth_model.predict(LSTM_training_inputs))+1) * training_set[pred_coin+'close'].values[:-window_len])[0], 
         label='Predicted')
ax1.set_title('Training Set: Single Timepoint Prediction')
ax1.set_ylabel('Ethereum Price ($)',fontsize=12)
ax1.legend(bbox_to_anchor=(0.15, 1), loc=2, borderaxespad=0., prop={'size': 14})
ax1.annotate('MAE: %.4f'%np.mean(np.abs((np.transpose(eth_model.predict(LSTM_training_inputs))+1)-\
            (training_set[pred_coin+'close'].values[window_len:])/(training_set[pred_coin+'close'].values[:-window_len]))), 
             xy=(0.75, 0.9),  xycoords='axes fraction',
            xytext=(0.75, 0.9), textcoords='axes fraction')
# figure inset code taken from http://akuederle.com/matplotlib-zoomed-up-inset
axins = zoomed_inset_axes(ax1, 3.35, loc=10) # zoom-factor: 3.35, location: centre
axins.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,5,9]])
axins.plot(model_data[model_data['date_format']< split_date]['date_format'][window_len:].astype(datetime.datetime),
         training_set[pred_coin+'close'][window_len:], label='Actual')
axins.plot(model_data[model_data['date_format']< split_date]['date_format'][window_len:].astype(datetime.datetime),
         ((np.transpose(eth_model.predict(LSTM_training_inputs))+1) * training_set[pred_coin+'close'].values[:-window_len])[0], 
         label='Predicted')
axins.set_xlim([datetime.date(2017, 3, 1), datetime.date(2017, 5, 1)])
axins.set_ylim([10,60])
axins.set_xticklabels('')
mark_inset(ax1, axins, loc1=1, loc2=3, fc="none", ec="0.5")
plt.show()


