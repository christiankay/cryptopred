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
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    LSTM_training_inputs.append(temp_set)
    
LSTM_training_outputs = (training_set['eth_close'][window_len:].values/training_set['eth_close'][:-window_len].values)-1


LSTM_test_inputs = []
for i in range(len(test_set)-window_len):
    temp_set = test_set[i:(i+window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    LSTM_test_inputs.append(temp_set)
LSTM_test_outputs = (test_set['eth_close'][window_len:].values/test_set['eth_close'][:-window_len].values)-1


print("example of input data", LSTM_training_inputs[0])

# I find it easier to work with numpy arrays rather than pandas dataframes
# especially as we now only have numerical data
LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)

LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)


# import the relevant Keras modules
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout

def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(output_dim=output_size, activation='linear',input_shape=(inputs.shape[1], inputs.shape[2]),return_sequences=False))
    
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model


# random seed for reproducibility
np.random.seed(202)
# initialise model architecture
eth_model = build_model(LSTM_training_inputs, output_size=1, neurons = 20)
# model output is next price normalised to 10th previous closing price
LSTM_training_outputs = (training_set['eth_close'][window_len:].values/training_set['eth_close'][:-window_len].values)-1
# train model on data
# note: eth_history contains information on the training error per epoch
eth_history = eth_model.fit(LSTM_training_inputs, LSTM_training_outputs, 
                            epochs=50, batch_size=1, verbose=2, shuffle=True,  validation_data=(LSTM_training_inputs, LSTM_training_outputs))



fig, ax1 = plt.subplots(1,1)

ax1.plot(eth_history.epoch, eth_history.history['loss'])
ax1.set_title('Training Error')

if eth_model.loss == 'mae':
    ax1.set_ylabel('Mean Absolute Error (MAE)',fontsize=12)
# just in case you decided to change the model loss calculation
else:
    ax1.set_ylabel('Model Loss',fontsize=12)
ax1.set_xlabel('# Epochs',fontsize=12)
plt.show()


eth_model.save('test_50ep.h5')
print ('model successfully saved!')

from keras.models import Model
import matplotlib.pyplot as plt


### print the keys contained in the history object
print(eth_history.history.keys())

### plot the training and validation loss for each epoch
plt.plot(eth_history.history['loss'])
plt.plot(eth_history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()