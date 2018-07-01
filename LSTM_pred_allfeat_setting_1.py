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
from sklearn.preprocessing import OneHotEncoder


def create_coin_dataset(coin_list, starttime = '2018-06-05', split_date = '2018-06-05' , features = ['_close','_volume','_close_off_high','_volatility', '_date']):
    
    print("Creating coin data set..")
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
     
#        kwargs = { coin+'_date': lambda x: (x[coin+'_date']- x[coin+'_date'][0])}
#        coin_market_info = coin_market_info.assign(**kwargs)
#        

            
        coin_market_info.rename(columns={coin+'_date_format': 'date_format'}, inplace=True)
        if market_info_prev is not False:
            market_info = pd.merge(market_info_prev,coin_market_info, on=['date_format'])
            market_info_prev = market_info
        else:
            market_info_prev = coin_market_info
            market_info = coin_market_info
            
            
            
   
        

    
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


def merge_lstm_data(starttime = '2016-06-24',  split_date = '2018-06-20', period = 900, features = ['_volatility', '_date','_close',
                                                              '_len_day_data' , '_neg_res', '_neut_tweet', '_pos_tweet']):
    
    filestr =  'start_'  + str(starttime) + '_split_' + str(split_date) + '_period_' + str(period)    
    try:
        
        training_set = pd.read_csv('CurDat/temp_results_train_'+filestr+'.csv')
        test_set = pd.read_csv('CurDat/temp_results_train_'+filestr+'.csv')
        model_data = pd.read_csv('CurDat/temp_results_model_'+filestr+'.csv')
        print('Loaded data from CurDat/temp_results_train_'+filestr+'.csv!')
        print('Loaded data from CurDat/temp_results_test_'+filestr+'.csv!')
        print('Loaded data from CurDat/temp_results_model_'+filestr+'.csv!')   
    except:    
        
        ####### important parameters##################
        
        print("Merge coin & twit data sets..")
        selected_coins = ['BTC', 'ETH']   #['BTC', 'LTC', 'ETH', 'XMR']
       # period = 900#86400 candlestick period in seconds; valid values are 300, 900, 1800, 7200, 14400, and 86400),
       # split_date = '2018-06-20' 
       # window_len = 25
        #######################################        
        df_ini = data_reader.get_poloniex_data(startdate = starttime)
        coin_list = df_ini.fetch_coin_data(selected_coins, period)   
        
        import raspi_tweet_fetcher.Load_Tweets_Class as LTC    
              
               ### init 
        get_tweet_data = LTC.get_tweets()
        ### fetch data based on query word
        #get_tweet_data.fetch_tweets(query='BITCOIN', count=100, pages=1)
        
        
        #get_tweet_data.fetch_stocktwits(query='BITCOIN')
        ### delete duplicates data from CSV
        tweet_data = get_tweet_data.read_and_clean_data_from_csv(query='BITCOIN')
        #data = get_tweet_data.data
                
        fin_data = coin_list['BTC']
        
        ### analyze sentiment
        tweet_results = get_tweet_data.analyze_Tweets(tweet_data)
        
        fin_data['Date'] = pd.to_datetime(fin_data.date_format)
        tweet_results['Date'] = pd.to_datetime(tweet_results['days'])
        
        merged = pd.merge(fin_data, tweet_results, how='outer', on='Date')
        merged.dropna(how = 'any' , inplace = True)
        
         
        new_coin_list = {}
        new_coin_list['BTC'] =   merged    
        
        training_set, test_set, model_data  = create_coin_dataset(new_coin_list, starttime = starttime,
                                                      split_date = split_date , 
                                                      features = ['_volatility', '_date','_close',
                                                                  '_len_day_data' , '_neg_res', '_neut_tweet', '_pos_tweet'])# '_date','_close_off_high'])    
        #
        #                                              features = ['_close','_volume','_volatility','_date', '_weightedAverage' , 
        #                                                          '_len_day_data' , '_neg_res', '_neut_tweet', '_pos_tweet'])
        
    
        
        panda_train = pd.DataFrame(training_set)  
        panda_train.to_csv('CurDat/temp_results_train_'+filestr+'.csv')
        panda_test = pd.DataFrame(test_set)  
        panda_test.to_csv('CurDat/temp_results_test_'+filestr+'.csv') 
        panda_model = pd.DataFrame(model_data)  
        panda_model.to_csv('CurDat/temp_results_model_'+filestr+'.csv') 
        
    return training_set, test_set, model_data
     


def prep_lstm_input(training_set, test_set, window_len):
    
    #### create windowed data  
    
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
    
##    ## scaling
    LSTM_training_inputs_scaled = np.zeros_like(LSTM_training_inputs)
    for i in range(LSTM_training_inputs.shape[2]):
        print(i)
        LSTM_training_inputs_scaled[:,:,i] = MaxAbsScaler().fit_transform(LSTM_training_inputs[:,:,i])
##        
    LSTM_test_inputs_scaled = np.zeros_like(LSTM_test_inputs)     
    for i in range(LSTM_test_inputs.shape[2]):
        LSTM_test_inputs_scaled[:,:,i]= MaxAbsScaler().fit_transform(LSTM_test_inputs[:,:,i])    
#LSTM_training_inputs = MaxAbsScaler().transform(LSTM_training_inputs)
#LSTM_test_inputs = MaxAbsScaler().transform(LSTM_training_inputs)

    return LSTM_training_inputs_scaled, LSTM_test_inputs_scaled



def prep_lstm_output_classification(training_data_set, test_data_set, window_len, accumulate):

    # model output is next price normalised to 10th previous closing price
    
    enc = OneHotEncoder()    
    test_set_close = test_data_set['BTC_close'].values  
    pred = [] 
    
    
    for i in range(len(test_set_close)-window_len):
        quo = []
        count = 0
        # accumulate next x values for prediction
        while count < accumulate:
            current= test_set_close[i+count]
            nxt = test_set_close[i+1+count]
            diff = nxt - current
            quo.append(diff/current)
            count = count + 1
            ## add label depanding on 
        if sum(quo)> 0.001:
            pred.append(2)
        elif sum(quo)  < -0.001:
            pred.append(1) 
        else:
            pred.append(0)
        
    LSTM_test_outputs = pred
    LSTM_test_outputs = np.asarray(LSTM_test_outputs)
    LSTM_test_outputs_onehot = enc.fit_transform(LSTM_test_outputs.reshape(-1,1)).toarray()
    
    class1  = (LSTM_test_outputs_onehot[:,0] == 1).sum()
    print("test data in class1: ", class1)
    class2  = (LSTM_test_outputs_onehot[:,1] == 1).sum()
    print("test data in class2: ", class2)
    class3  = (LSTM_test_outputs_onehot[:,2] == 1).sum()
    print("test data in class3: ", class3)
    
    training_set_close = training_data_set['BTC_close'].values  
    pred = [] 
    quo = []
    for i in range(len(training_set_close)-window_len):
        quo = []
        count = 0
        # accumulate next x values for prediction
        while count < accumulate:
            current= training_set_close[i+count]
            nxt = training_set_close[i+1+count]
            diff = nxt - current
            quo.append(diff/current)
            count = count + 1
            ## add label depanding on profitability
        if sum(quo)> 0.001:
            pred.append(2)
        elif sum(quo)  < -0.001:
            pred.append(1) 
        else:
            pred.append(0)
        
            
    LSTM_training_outputs = pred
    LSTM_training_outputs = np.asarray(LSTM_training_outputs)
    LSTM_training_outputs_onehot = enc.fit_transform(LSTM_training_outputs.reshape(-1,1)).toarray()
    
    class1  = (LSTM_training_outputs_onehot[:,0] == 1).sum()
    print("training data in class1: ", class1)
    class2  = (LSTM_training_outputs_onehot[:,1] == 1).sum()
    print("training data in class2: ", class2)
    class3  = (LSTM_training_outputs_onehot[:,2] == 1).sum()
    print("training data in class3: ", class3)
    
    return LSTM_training_outputs_onehot, LSTM_test_outputs_onehot

def prep_lstm_output_reg(training_data_set, test_data_set):

    # model output is next price normalised to 10th previous closing price
    
    enc = OneHotEncoder()    
    test_set_close = test_data_set['BTC_close'].values  
    pred = [] 
    
    
    for i in range(len(test_set_close)-window_len):
           
            pred.append(test_set_close[i+1])
        
    LSTM_test_outputs = pred
    LSTM_test_outputs = np.asarray(LSTM_test_outputs)

    

    
    training_set_close = training_data_set['BTC_close'].values  
    pred = [] 
  
    for i in range(len(training_set_close)-window_len):
            
            pred.append(training_set_close[i+1])
       
            
    LSTM_training_outputs = pred
    LSTM_training_outputs = np.asarray(LSTM_training_outputs)

    

    
    return LSTM_training_outputs, LSTM_test_outputs

def load_data(data, seq_len, normalise_window):

    
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        try:
            normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
            normalised_data.append(normalised_window)
        except:
            print('Normalize failed')
            normalised_window = [0.0 for p in window]
            normalised_data.append(normalised_window)
            
    return normalised_data

    
def build_model(layers):
    # import the relevant Keras modules
    from keras.models import Sequential
    from keras.layers import Activation, Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
   
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    ###loss="mse", optimizer="rmsprop"
    ###loss='categorical_crossentropy', optimizer="adam"
    print ("Compilation Time : ", time.time() - start)
    return model


def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[np.newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
   
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()
    
    
if __name__ is "__main__":
    
    plt.close('all')
    
    window_len = 50
    period = 900
    ##### get data
    


    training_set, test_set, model_data = merge_lstm_data(starttime = '2018-06-14', split_date = '2018-06-25', period = period, features = ['_volatility', '_date','_close',
                                                              '_len_day_data' , '_neg_res', '_neut_tweet', '_pos_tweet'])
    
        
        
        
        
    model_data['Date'] = pd.to_datetime(model_data['date_format'], errors = 'coerce')
    
    data_frame_res = pd.DataFrame( index=model_data['Date'])
    #data_frame_res = data_frame_res.drop('days', axis=1)
    for col in list(model_data):
        
        
        data_frame_res[col] = pd.Series(data=model_data[col].values, index=model_data['Date'])
    

         
#    results.plot()
#        
#    AO['1980-05':'1981-03'].plot()   
    data_frame_res.plot(subplots=True) 
  #  data_frame_res.plot(subplots=False) 
   # AO_mm = data_frame_res.resample("H").mean()
  #  AO_mm.plot(subplots=True) 


#    ################ get fin data only#### 
#    selected_coins = ['BTC', 'ETH']   #['BTC', 'LTC', 'ETH', 'XMR']     
#    df_ini = data_reader.get_poloniex_data(startdate = '2018-05-28')
#    coin_list = df_ini.fetch_coin_data(selected_coins, period)       
#    fin_data = coin_list['BTC']
    
  #  [x_train, y_train, x_test, y_test] = load_data(np.asarray(model_data['BTC_close']), 12, True)
    
  ### good one starttime = '2018-06-14', split_date = '2018-06-25', period = 900
    ##[x_train, y_train, x_test, y_test] = load_data(np.asarray(model_data['BTC_close']), 12, True)
    
    #features = ['BTC_volatility', 'BTC_date','BTC_close','BTC_len_day_data', 'BTC_neg_res', 'BTC_neut_tweet', 'BTC_pos_tweet']
    features = ['BTC_volatility','BTC_close','BTC_len_day_data', 'BTC_neg_res', 'BTC_neut_tweet', 'BTC_pos_tweet']
    x_train = None
    for feature in features:
        print(feature)
        if x_train is None:
            [x_train, y_train, x_test, y_test] = load_data(np.asarray(model_data[feature]), 12, True)
            
        else:    
            [x_trainb, y_trainb, x_testb, y_testb] = load_data(np.asarray(model_data[feature]), 12, True)
            x_train = np.concatenate((x_train, x_trainb), axis = 2)
            x_test = np.concatenate((x_test, x_testb), axis = 2)
   
    [x_trainoff, y_train, x_testoff, y_test] = load_data(np.asarray(model_data['BTC_close']), 12, True)
    
    datPath = 'models/'
    path = os.path.join(datPath, 'BTC_' + str(period) + '_' + str(window_len) + '.h5')
    
    pred_coin = 'BTC_'
    try:
    
        model = load_model(path)
        eth_model = model
    
        print("Model succesfully loaded")
    except:
        print("Could not find saved model data!!")
        print("Start training model!")
            # random seed for reproducibility
        
        np.random.seed(7)
        # initialise model architecture
        eth_model = build_model(layers = [x_train.shape[2],20,50,1])
        # train model on data
        # note: eth_history contains information on the training error per epoch
        eth_history = eth_model.fit(x_train, y_train, 
                                    epochs=20, batch_size=128, verbose=1, shuffle=True,  validation_data=(x_test, y_test))
        
        
        if not os.path.exists(datPath):
            os.mkdir(datPath)
            eth_model.save(path)
            
            print ('model successfully saved!')

    
    
    y_pred = eth_model.predict(x_test)
    
    enc = OneHotEncoder()
    
    y_pred_one = enc.fit_transform(y_pred.reshape(-1,1)).toarray()
    
    #target_names = ['class 0', 'class 1', 'class 2']
    #print(classification_report(y_test, y_pred_one, target_names=target_names))
    
    
    #Predict sequence of 10 steps before shifting prediction run forward by 10 steps
    prediction_len = 8
    window_size = 8
    
    predictions = predict_sequences_multiple(eth_model, x_test, window_size, prediction_len)
    plot_results_multiple(predictions, y_test, 8)
    
    
    



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
    
    
    
    
    
    
    

######
#from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
#from mpl_toolkits.axes_grid1.inset_locator import mark_inset
#
#fig, ax1 = plt.subplots(1,1)
#ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,5,9]])
#ax1.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,5,9]])
#ax1.plot(model_data[model_data['date_format']< split_date]['date_format'][window_len:].astype(datetime.datetime),
#         training_set[pred_coin+'close'][window_len:], label='Actual')
#ax1.plot(model_data[model_data['date_format']< split_date]['date_format'][window_len:].astype(datetime.datetime),
#         ((np.transpose(eth_model.predict(LSTM_training_inputs))+1) * training_set[pred_coin+'close'].values[:-window_len])[0], 
#         label='Predicted')
#ax1.set_title('Training Set: Single Timepoint Prediction')
#ax1.set_ylabel('Ethereum Price ($)',fontsize=12)
#ax1.legend(bbox_to_anchor=(0.15, 1), loc=2, borderaxespad=0., prop={'size': 14})
#ax1.annotate('MAE: %.4f'%np.mean(np.abs((np.transpose(eth_model.predict(LSTM_training_inputs))+1)-\
#            (training_set[pred_coin+'close'].values[window_len:])/(training_set[pred_coin+'close'].values[:-window_len]))), 
#             xy=(0.75, 0.9),  xycoords='axes fraction',
#            xytext=(0.75, 0.9), textcoords='axes fraction')
## figure inset code taken from http://akuederle.com/matplotlib-zoomed-up-inset
#axins = zoomed_inset_axes(ax1, 3.35, loc=10) # zoom-factor: 3.35, location: centre
#axins.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,5,9]])
#axins.plot(model_data[model_data['date_format']< split_date]['date_format'][window_len:].astype(datetime.datetime),
#         training_set[pred_coin+'close'][window_len:], label='Actual')
#axins.plot(model_data[model_data['date_format']< split_date]['date_format'][window_len:].astype(datetime.datetime),
#         ((np.transpose(eth_model.predict(LSTM_training_inputs))+1) * training_set[pred_coin+'close'].values[:-window_len])[0], 
#         label='Predicted')
#axins.set_xlim([datetime.date(2017, 3, 1), datetime.date(2017, 5, 1)])
#axins.set_ylim([10,60])
#axins.set_xticklabels('')
#mark_inset(ax1, axins, loc1=1, loc2=3, fc="none", ec="0.5")
#plt.show()
#

