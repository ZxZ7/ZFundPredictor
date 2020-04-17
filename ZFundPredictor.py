import math
from datetime import datetime, timedelta, date
import re

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



class FundPredictor:
    def __init__(self, funds):
        self.funds = funds     # historical data of all funds
        self.ticker = ''       # ticker of the fund to be predicted
        self.fund_allfeat = [] # data of the fund (to be precited) with all features
        self.fund = []         # data of the fund (to be precited) with predicting features
        self.ma = None         # x-day exponential moving averages used for prediction (e.g. ma=[20,50])
        self.log_form = True   # when True, use exponential form of the historical prices
        self.indicators = 'all' # Technical indicators in use
    
        sns.set_style("darkgrid")
        sns.set_context("notebook")
        

    def get_features(self, ticker, log_form=True, ma=None, indicators='all', show_graph=False):
        '''
        【Parameters】
            `log_form`:   when True, use exponential form of the historical prices.
            `ma`:         moving average periods. Either None, or a two-item list.
            `show_graph`: if True, graph the historical performance of the fund,
                          as well as its features.
            `indicator`:  `all` - adopt all available indicators;
                          `ema` - adopt only EMA indictors.
        
        Available time series features (based on daily data):
            `diff`:       x-order (absolute) differencing of historical prices
                          e.g. `diff2` = price_t - price_(t-2).
            `r`:          daily return of the fund.
            `sindex_r`:   market daily return.
            `r_premium`:  market adjusted daily return of the fund.
            `EMA`:        x-day exponential moving averages used for prediction
                          e.g. ma=[20,50] will generate two varables -
                              `ema20`, the 20-day EMA, and `ema50`, the 50-day EMA.
            `dist_EMA`:   the distances between fund price and x-day EMA of the fund on each day.
            `signal_EMA`: one-time signal. `-1` if the shorter EMA crosses above the longer
                          EMA on that day, `1` vice versa, and `0` if there is no cross.
            `BB`:         the Bollinger Bands indicators over the past x days.
                          `lower_BB` means the lower band, and `upper_BB` for the upper.
            `status_BB`:  short-term status. `-1` if the value of BB indicator is below the
                          lower band, `1` if it is above the upper band, and `0` otherwise.
        '''
        assert indicators in ['all', 'ema'], "`indicators` must in ['all', 'ema']"
        assert ma == None or len(ma) == 2, "`ma` must contain two periods."
        
        self.indicators = indicators
        self.ticker = ticker
        self.ma = ma
        if log_form != self.log_form:
            self.log_form = log_form
                
        fund = pd.DataFrame(self.funds[ticker]).assign(**{
#                             'diff1': lambda df: df[ticker].diff(1),
#                             'diff2': lambda df: df[ticker].diff(2),
                            'r': lambda df: (df[ticker] - df[ticker].shift(1))/df[ticker].shift(1)*100})

        fund = pd.concat([fund, self.funds['sindex_r']], axis=1)
        fund['r_premium'] = fund['r'] - fund['sindex_r']

        keep_col = [ticker, 'diff1', 'diff2', 'sindex_r', 'r_premium']
        if ma:
            for x in ma:
                keep_col.append('dist_EMA%s'%x)
#                 keep_col.append('EMA%s'%x)
                fund['EMA%s'%x] = fund[ticker].ewm(span=x, adjust=False).mean()
#                 fund['dist_EMA%s'%x] = (fund[ticker] - fund['EMA%s'%x])/fund['EMA%s'%x]*100
                fund['dist_EMA%s'%x] = fund[ticker] - fund['EMA%s'%x]
                fund.loc[fund[:x-1].index ,'dist_EMA%s'%x] = None

                fund['SMA%s'%x] = fund[ticker].rolling(window=x).mean()
                rolling_std  = fund[ticker].rolling(window=x).std()
                fund['upper_BB%s'%x] = fund['SMA%s'%x] + (rolling_std*2)
                fund['lower_BB%s'%x] = fund['SMA%s'%x] - (rolling_std*2)
                
                if indicators == 'all':
                    keep_col.append('status_BB%s'%x)
                    fund_BB = (fund[ticker] - fund['SMA%s'%x]) / (2*rolling_std)
    # #                 fund['status_BB%s'%x] = fund_BB.apply(lambda x: x-1 if x > 1 
    # #                                                       else (x+1 if x < -1 else 0))
                    fund['status_BB%s'%x] = fund_BB.apply(lambda x: 1 if x > 1 
                                                          else (-1 if x < -1 else 0))
    
            if indicators == 'all':
                keep_col.append('signal_EMA')
                fund['signal_EMA'] = 0

                if ma[0] < ma[1]:
                    fund['diff_EMA'] = fund['EMA%s'%ma[0]] - fund['EMA%s'%ma[1]]
                else:
                    fund['diff_EMA'] = fund['EMA%s'%ma[1]] - fund['EMA%s'%ma[0]]
                fund['diff_lag'] = fund['diff_EMA'].shift(1)

                fund.loc[fund[(fund['diff_EMA'] < 0) & (fund['diff_lag'] >= 0)].index, 'signal_EMA'] = -1
                fund.loc[fund[(fund['diff_EMA'] > 0) & (fund['diff_lag'] <= 0)].index, 'signal_EMA'] = 1
                fund.drop(columns=['diff_lag'], inplace=True)

        
        if log_form:
            fund['original_price'] = fund[ticker]
            fund[ticker] = fund[ticker].apply(np.log)
        
        fund = fund.dropna()
        self.fund_allfeat = fund
        self.fund = fund[[i for i in keep_col if i in fund.columns]].dropna()
        
        
        # drop columns that only contain 0 values 
        for col in self.fund:
            if int((self.fund[col] == 0).mean()) == 1:
                self.fund.drop(columns=[col], inplace=True)
        
        if show_graph:
            self.draw_histories(ticker, ma, log_form)


    def draw_histories(self, ticker, ma=None, log_form=True):
        if ticker != self.ticker or ma != self.ma or log_form != self.log_form:
            self.get_features(ticker, log_form=log_form, ma=ma)
        fund = self.fund
        if log_form:
            fund_prices = fund.iloc[:, 0].apply(np.exp)
        else:
            fund_prices = fund.iloc[:, 0]
        
        corrmat = fund.corr()
        plt.subplots(figsize=(6,6))
        sns.heatmap(corrmat, vmax=0.9, square=True)
        
        
        fig = plt.figure(figsize=(16,10))

        ax1 = fig.add_subplot(411)
        ax1.plot(fund.index[:], fund_prices)
        ax1.set_ylabel('Price (daily)')
        plt.title(ticker, fontsize=14)

        ax2 = fig.add_subplot(412)
        ax2.plot(fund.index[:], fund['sindex_r'][:], label='market')
        ax2.plot(fund.index[:], fund['r_premium'][:], label='stock(mkt_adjusted)')
        ax2.set_ylabel('Return (daily)')
        ax2.legend()
        
        i = 2
        for var in ['diff2', 'diff1']:
            if var in fund.columns:
                i += 1
                ax = fig.add_subplot(4,1,i)
                ax.plot(fund.index[:], fund[var][:])            
                ax.set_ylabel(var)
        
        if ma:
            len_ = len(ma) + 1
            fig = plt.figure(figsize=(16, len_*5))
            
            ax = fig.add_subplot(len_, 1, 1)
            for x in ma:
                ax.plot(fund.index[:], fund['dist_EMA%s'%x][:], label='dist_EMA%s'%x)
#                 ax.plot(fund.index[:], self.fund_allfeat['EMA%s'%x][:], label='EMA%s'%x)
            ax.legend()
            ax.set_ylabel('Distance From EMA')
            
            for i in range(len(ma)):
                ax = fig.add_subplot(len_, 1, i+2)
                x = ma[i]
                ax.plot(fund.index[:], fund_prices, label='price')
                ax.plot(fund.index[:], self.fund_allfeat['SMA%s'%x][:], label='SMA%s'%x)
                ax.plot(fund.index[:], self.fund_allfeat['upper_BB%s'%x][:], label='upper_band%s'%x)
                ax.plot(fund.index[:], self.fund_allfeat['lower_BB%s'%x][:], label='lower_band%s'%x)
                ax.legend()
                ax.set_ylabel('Bollinger Bands')
    
    
    def split_data(self, lookback, lookahead, test_size=0.2):
        fund = self.fund
        
        date_ = fund.shape[0] - lookback - lookahead

        X = np.array([fund.iloc[i:i+lookback].values for i in range(date_+1)])
        y = np.array([fund.iloc[i+lookback:i+lookback+lookahead].values for i in range(date_+1)])

        X_latest = fund.iloc[date_+lookahead:date_+lookback+lookahead, :].values
        X_latest = X_latest.reshape(1, X_latest.shape[0], X_latest.shape[1]) # used to generate future prediction

        dates = np.array([str(i) for i in fund.iloc[lookback:].index])
        
        k = np.round(len(X)*(1-test_size)).astype(int)
        k_train = k - lookback - lookahead + 1  # to avoid train/text sets overlapping
        X_train = X[:k_train,:]
        X_test = X[k:,:,:]
        y_train = y[:k_train,:,0]
        y_test = y[k:,:,0]

        return [X_train, X_test, y_train, y_test, dates[k:], X_latest]


    def lstm_model(self, lookback, lookahead, in_feat, dropout=0.2, lr=0.0005):
        model = keras.Sequential()

        model.add(layers.LSTM(256, input_shape=(lookback, in_feat), return_sequences=True))
        model.add(layers.Dropout(dropout))

        model.add(layers.LSTM(256, input_shape=(lookback, in_feat), return_sequences=False))
        model.add(layers.Dropout(dropout))

        model.add(layers.Dense(32, activation='relu'))        
        model.add(layers.Dense(lookahead, activation='linear'))

        adam = keras.optimizers.Adam(lr=lr)
        
        model.compile(loss='mse', optimizer=adam, metrics=['mse'])
        return model


    def predict_score(self, model, X_train, y_train, X_test, y_test, X_latest):
        y_pred = model.predict(X_test)
        new_pred = model.predict(X_latest)
    
        trainScore = model.evaluate(X_train, y_train, verbose=0)[0]
        testScore = model.evaluate(X_test, y_test, verbose=0)[0]
    
        r2 = r2_score(y_test, y_pred)
        
        if self.log_form:
            y_pred = np.exp(y_pred)
            new_pred = np.exp(new_pred)

        return trainScore, testScore, r2, y_pred, new_pred


    def get_prediction(self, ticker, windows, log_form=True, model_type='lstm',
                       epochs=90, batches=300, dropout=0.2, lr=0.0005,
                       ma=None, indicators='all', callbacks=None, verbose=True, show_history=False):
        '''
        ** Main function **
        Get one single prediction per window, with respect to the hyperparameters provided.
        【Parameters】
         `windows`:      Must be a list of two-obj lists, each containing a lookback period
                         and a lookahead period.
                         i.e. [[lookback1, lookahead1], [lookback2, lookahead2],...]
                         `lookback` - historical days for prediction.
                         `lookahead` - future days to be predicted.
         `model_type`:   'lstm' with hyperparameters `epochs`, `batches`, `lr`, `dropout`, `callbacks`
         `verbose`:      if True, print and graph the predicted results.
         `show_history`: if True, graph the training history.
        【Returns】
         `self.preds`:   a list of dicts, each presented in the following form -
                                 {'window': prediction window,
                                 'data': the return values of function `split_data()`,
                                 'pred': the return values of function `predict_score()`,
                                 'model': prediction model,
                                 'history': training history}
        '''
        if ticker != self.ticker or ma != self.ma or indicators != self.indicators:
            self.get_features(ticker, ma=ma, indicators=indicators, log_form=log_form)
        fund = self.fund

        try:
            windows[0][0]
        except TypeError:
            windows = [windows] # in case of single model

        self.preds = []
        for window in windows:
            lookback, lookahead = window      

            data = self.split_data(lookback, lookahead)

            X_train, X_test, y_train, y_test, date_test, X_latest = data
            len_ = int(X_train.shape[0]*0.8)
            len_train = len_ - lookback - lookahead + 1  # to avoid train/validation sets overlapping
            X_tr = X_train[:len_train].copy()
            y_tr = y_train[:len_train].copy()
            X_valid = X_train[len_:].copy()
            y_valid = y_train[len_:].copy()
            
            if model_type == 'lstm':
                model = self.lstm_model(lookback, lookahead, X_train.shape[-1], dropout=dropout, lr=lr)

                train_history = model.fit(X_tr, y_tr, 
                                          epochs=epochs, batch_size=batches,
                                          callbacks=callbacks, verbose=0,
                                          validation_data=(X_valid, y_valid))
 
            
            pred = self.predict_score(model, X_train, y_train, X_test, y_test, X_latest)
            self.preds.append({'window':window, 'data':data, 'pred':pred,
                               'model':model, 'history':train_history})

        if verbose:
            self.allow_verbosity(model_type=model_type)
        if show_history:
            self.show_history()
        return self.preds

    
    
    def ensemble_prediction(self, ticker, ma_basket, dropout_basket, pred_params,
                            verbose=True, show_history=False):
        '''
        ** Main Function **
        Get one ensemble prediction per window, with fine-tuned hyperparameters and stacked models.
        Stacking strategy: averaging up to 3 predictions with the best R-squared scores.

        【Parameters】
         `ma_basket`:      a list of `ma` options used for fine-tuning.
         `dropout_basket`: a list of `dropout` options used for fine-tuning.
         `pred_params`:    a dict containing other parameters for function `get_prediction()`.
         `verbose`:        if True, print and graph the predicted results.
         `show_history`:   if True, graph the training history.
        【Returns】
         `fine_tuning`:    a dict of subdicts, each subdict corresponding to a model with specific window.
                           e.g. model_name == 'pred(50, 1)', meaning 50-day lookback, 1-day lookahead.
                           {model_name:{'r2': [r2_tuning_1, r2_tuning_2, ...],
                                        'preds': [y_pred_tuning_1, y_pred_tuning_2, ...],
                                        'newpreds': [...],
                                        `histories`: [...],
                                        `models`: [...],
                                        'stacked': [stacked_y_pred, stacked_future_pred] }}
                           For keys other than `stacked`, the size of their correpsonding
                           value lists should be (len(ma_basket)*len(dropout_basket)).
        '''
        
        self.stacking_process(indicators='all', ticker=ticker, ma_basket=ma_basket,
                              dropout_basket=dropout_basket, pred_params=pred_params)
        
        keys = predictor.fine_tuning.keys()
        r2_stacked = np.array([self.fine_tuning[key]['stacked']['r2'] for key in keys])
        
        # if 'all' indicators do not perform well, try using only 'ema' indicators
        if r2_stacked.mean()<0.63 or r2_stacked[-1]<0.5 or r2_stacked.min()<0:
            fune_tuning_ = self.fine_tuning
            
            self.stacking_process(indicators='ema', ticker=ticker, ma_basket=ma_basket,
                                  dropout_basket=dropout_basket, pred_params=pred_params)            
            
            r2_stacked_new = np.array([self.fine_tuning[key]['stacked']['r2'] for key in keys])
            
            r2_diff = r2_stacked - r2_stacked_new
            if (r2_diff>0.01).mean() >= 0.5 and (r2_diff<-0.05).sum() == 0:
                self.fine_tuning = fune_tuning_
                
        if verbose:
            self.allow_verbosity(fine_tuning=True)
        if show_history:
            self.show_history(fine_tuning=True)            
            
        return self.fine_tuning
    
    
    
    def stacking_process(self, ticker, ma_basket, dropout_basket, pred_params,
                         indicators='all'):
        '''
        The model stacking process for ensemble prediction.
        '''
        self.fine_tuning = {}
        self.hypers = []
        for ma in ma_basket:
            for dropout in dropout_basket:
                self.hypers.append({'ma':ma, 'dropout':dropout})
                predicted = self.get_prediction(ticker, ma=ma, dropout=dropout,
                                                indicators=indicators,
                                                verbose=False, **pred_params)
                for i in range(len(predicted)):
                    model_name = '(%s,%s)'% (predicted[i]['window'][0], predicted[i]['window'][1])
                    if model_name not in self.fine_tuning:
                        self.fine_tuning[model_name] = {'r2':[], 'preds':[], 'newpreds':[], 'histories':[], 'models':[]}
                    
                    self.fine_tuning[model_name]['r2'].append(predicted[i]['pred'][2])
                    self.fine_tuning[model_name]['preds'].append(predicted[i]['pred'][-2])
                    self.fine_tuning[model_name]['newpreds'].append(predicted[i]['pred'][-1])
                    self.fine_tuning[model_name]['histories'].append(predicted[i]['history'])
                    self.fine_tuning[model_name]['models'].append(predicted[i]['model'])
        
        
        # Note: According to our splitting method, different ma values lead to 
        #       different data length, which in turn results in different length
        #       of prediction. In order to perform model stacking, we will need
        #       equal length accross all predictions. The process below is intended
        #       to find the maximum possible length given the `ma_basket`.
        max_ma = np.array([i for ma in ma_basket if ma != None for i in ma]).max()
        data_length = self.funds[ticker].shape[0] - max_ma + 1
        
        test_size = 0.2
        for i, key in enumerate(self.fine_tuning.keys()):
            lookback, lookahead = self.preds[i]['window']
            max_data_len = data_length - lookback - lookahead + 1
            max_test_len = max_data_len - np.round(max_data_len*(1-test_size)).astype(int)
                 
            model = self.fine_tuning[key]
            
            for num in range(len(model['preds'])):
                if len(model['preds'][num]) > max_test_len:
                    model['preds'][num] = model['preds'][num][-max_test_len:]
            

            ## stacking models ##
            model['stacked'] = {'hypers':[]}
            pred_sorted = np.array(model['r2']).argsort()  # returns the indices that would sort the array
            
            model['stacked']['hypers'].append(self.hypers[pred_sorted[-1]])
            
            if model['r2'][pred_sorted[-1]] - model['r2'][pred_sorted[-2]] <= 0.03:
                model['stacked']['hypers'].append(self.hypers[pred_sorted[-2]])
                y_pred = np.average([model['preds'][pred_sorted[-1]], model['preds'][pred_sorted[-2]]], axis=0)
                fut_pred = np.average([model['newpreds'][pred_sorted[-1]], model['newpreds'][pred_sorted[-2]]], axis=0)
                                
                if len(pred_sorted) > 3 and model['r2'][pred_sorted[-2]] - model['r2'][pred_sorted[-3]] <= 0.01:
                    model['stacked']['hypers'].append(self.hypers[pred_sorted[-3]])
                    y_pred = np.average([y_pred, model['preds'][pred_sorted[-3]]], axis=0, weights=[2/3,1/3])
                    fut_pred = np.average([fut_pred, model['newpreds'][pred_sorted[-3]]], axis=0, weights=[2/3,1/3])

            else:
                y_pred = model['preds'][pred_sorted[-1]]
                fut_pred = model['newpreds'][pred_sorted[-1]]
            
            model['stacked']['preds'] = y_pred
            model['stacked']['newpreds'] = fut_pred
            model['stacked']['r2'] = r2_score(np.exp(self.preds[i]['data'][3]), y_pred)

    
    def allow_verbosity(self, model_type='lstm', fine_tuning=False, show_period=120):
            
        fig, ax = plt.subplots(figsize=(16,5))
        y = []
        date = []
        future = []
        for i, pred in enumerate(self.preds):
            y_test = pred['data'][3]
            date_test = pred['data'][4]
            future_dates = ['T + %s'% (x+1) for x in range(pred['window'][1])]
            date_test_fut = list(date_test[-show_period:]) + future_dates
            
            if self.log_form:
                y_test = np.exp(y_test)
            if len(y_test) > len(y):
                y = y_test
                date = date_test
            if len(future_dates) > len(future):
                future = future_dates

            print('='*15, "%s / Model (%s, %s) / '%s'"% (self.ticker,
                   pred['window'][0], pred['window'][1], self.indicators),'='*15)
            print('Train Size:', pred['data'][0].shape, pred['data'][1].shape)
            print('Test Size:', pred['data'][2].shape, pred['data'][3].shape)
            print('Size of Data for Future Prediction:', pred['data'][-1].shape)
            
            
            if fine_tuning:     # when called by ensemble_prediction()
                key = [k for k in self.fine_tuning.keys()][i]
                stacked_preds = self.fine_tuning[key]['stacked']
                                
                print('Selected Hyperparameters:', stacked_preds['hypers'])
                print('R-Squared (ensemble): %.4f' % stacked_preds['r2'])
                print('Future Prediction: %s' % [round(float(x),4) for x in stacked_preds['newpreds'][0]])
                
                pred_prices = [p[-1] for p in stacked_preds['preds']]
                pred_prices = pred_prices[-show_period:] + list(stacked_preds['newpreds'][0])
                ax.plot(date_test_fut, pred_prices, label='Ensemble '+key)

            else:     # when called by get_prediction()

                if model_type == 'lstm':
                    print('Stopped at epoch: %s' % pred['history'].epoch[-1])
                    print('Train Score: %.5f MSE (%.2f RMSE)' % (pred['pred'][0], math.sqrt(pred['pred'][0])))
                    print('Test Score: %.5f MSE (%.2f RMSE)' % (pred['pred'][1], math.sqrt(pred['pred'][1])))

                print('R-Squared: %.4f' % pred['pred'][2])
                print('Future Prediction: %s' % [round(float(i),4) for i in pred['pred'][-1][0]])
                
                pred_prices = [p[-1] for p in pred['pred'][-2]]
                pred_prices = pred_prices[-show_period:] + list(pred['pred'][-1][0])
                ax.plot(date_test_fut, pred_prices,
                        label='Model (%s,%s)'% (pred['window'][0], pred['window'][1]))
        
        if y.shape[1] > 1:
            y = np.array([p[0] for p in y[:-1]]+list(y[-1]))
        ax.plot(date[-show_period:], y[-show_period:], label='Actual')
        
        all_dates = list(date)[-show_period:] + future
        plt.xticks(range(0,len(all_dates),2), all_dates[::2], rotation=90)
        ax.vlines(date_test[-1], y[-show_period:].min()-0.01, y[-show_period:].max()+0.01,
                  linestyles='dashed', colors='tab:pink')
        
        txt = 'Note: For Model (b,a), the plotted value of any past date indicates the \
price prediction for that day using historical quotes from (t-b-a) to (t-a).\n\
The plotted values for future dates (T+1,...) are `a`-day consecutive forecast(s) from the lastest predicting period \
(i.e. from (T-b-a) to (T-a), where T is the last historical date).'
        fig.text(.1, -.15, txt, ha='left')
        plt.ylabel(self.ticker, fontsize=13)
        plt.legend()
        plt.show()
        
        
    def show_history(self, fine_tuning=False):
        if fine_tuning:
            len_ = len(self.fine_tuning)
            fig = plt.figure(figsize=(3*len(self.hypers), 2*len_))
            j = 0
            for key in predictor.fine_tuning.keys():
                for history in self.fine_tuning[key]['histories']:
                    ax = fig.add_subplot(len_, len(self.hypers), j+1)
                    fig.tight_layout()
                    ax.plot(history.history['loss'])
                    ax.plot(history.history['val_loss'])

                    if j < len(self.hypers):
                        plt.title('MA: {}, Dropout: {}'.format(self.hypers[j]['ma'], self.hypers[j]['dropout']))
                    if j % len(self.hypers) == 0:
                        plt.ylabel('Model '+key)
                    if j >= len(self.hypers)*(len_-1):
                        plt.xlabel('epoch')
                    plt.legend(['train loss', 'val loss'], loc='upper right')

                    j += 1
        
        else:
            fig = plt.figure(figsize=(4*len(self.preds), 3))
            for j,pred in enumerate(self.preds):
                ax = fig.add_subplot(1, len(self.preds), j+1)
                fig.tight_layout()
                
                ax.plot(pred['history'].history['loss'])
                ax.plot(pred['history'].history['val_loss'])
                plt.xlabel('epoch')
                plt.title(f'Model (%s,%s)'% (pred['window'][0], pred['window'][1]))
                plt.legend(['train loss', 'val loss'], loc='upper right')
        
        plt.show()