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
        self.ema = None        # x-day exponential moving averages used for prediction (e.g. ema=[20,50])
        self.log_form = True   # when True, use exponential form of the historical prices

        sns.set_style("darkgrid")
        sns.set_context("notebook")
    
    
    def get_features(self, ticker, log_form=True, ema=None):
        '''
        Available time series features (based on daily data):
            `diff1`/`diff2`: x-order (absolute) differencing of historical prices
                             e.g. diff2 = price_t - price_(t-2)
            `r`: daily return of the fund
            `sindex_r`: market daily return
            `r_premium`: market adjusted daily return of the fund
            `log_form`: when True, use exponential form of the historical prices
            `ema`: x-day exponential moving averages used for prediction
                   e.g. ema=[20,50] will generate two varables -
                        `ema20`, the 20-day EMA, and `ema50`, the 50-day EMA
            `dist_ema`: the distances between fund price and the fund's x-day EMA on each day
        '''
        self.ticker = ticker
        self.ema = ema
        if log_form != self.log_form:
            self.log_form = log_form
                
        fund = pd.DataFrame(self.funds[ticker]).assign(**{
#                             'diff1': lambda df: df[ticker].diff(1),
#                             'diff2': lambda df: df[ticker].diff(2),
                            'r': lambda df: (df[ticker] - df[ticker].shift(1))/df[ticker].shift(1)*100})

        fund = pd.concat([fund, self.funds['sindex_r']], axis=1)
        fund['r_premium'] = fund['r'] - fund['sindex_r']

        keep_col = [ticker, 'diff1', 'diff2', 'sindex_r', 'r_premium']
        if ema:
            for x in ema:
                keep_col.append('dist_EMA%s'%x)
#                 keep_col.append('EMA%s'%x)
                fund['EMA%s'%x] = fund[ticker].ewm(span=x, adjust=False).mean()
#                 fund['dist_EMA%s'%x] = (fund[ticker] - fund['EMA%s'%x])/fund['EMA%s'%x]*100
                fund['dist_EMA%s'%x] = fund[ticker] - fund['EMA%s'%x]

        if log_form:
            fund['ori_price'] = fund[ticker]
            fund[ticker] = fund[ticker].apply(np.log)

        self.fund_allfeat = fund.dropna()
        self.fund = fund[[i for i in keep_col if i in fund.columns]].dropna()


    def draw_histories(self, ticker, ema=None, log_form=True):
        if ticker != self.ticker or ema != self.ema or log_form != self.log_form:
            self.get_features(ticker, log_form=log_form, ema=ema)
        fund = self.fund
        
        corrmat = fund.corr()
        plt.subplots(figsize=(10,5))
        sns.heatmap(corrmat, vmax=0.9, square=True)

        fig = plt.figure(figsize=(10,10))

        ax1 = fig.add_subplot(411)
        if log_form:
            ax1.plot(fund.index[:], fund.iloc[:, 0].apply(np.exp))
        else:
            ax1.plot(fund.index[:], fund.iloc[:, 0])
        ax1.set_ylabel('Price (daily)')
        plt.title(fund.columns[0])

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

        if ema:
            fig, ax = plt.subplots(figsize=(10,5))
            for x in ema:
                ax.plot(fund.index[:], fund['dist_EMA%s'%x][:], label='dist_EMA%s'%x)
#                 ax.plot(fund.index[:], fund['EMA%s'%x][:], label='EMA%s'%x)
            ax.legend()


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
        
        model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
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
                       ema=None, callbacks=None, verbose=True, show_history=False):
        '''
        ** Main function **
        Get one single prediction per window, with respect to the hyperparameters provided.
        【Parameters】
            `windows`: periods for predicting and periods to be predicted.
                       Must be a list of two-obj lists, each containing a lookback period and a lookahead period.
                       i.e. [[lookback1, lookahead1], [lookback2, lookahead2],...]
            `model_type`: 'lstm' with hyperparameters `epochs`, `batches`, `lr`, `dropout`, `callbacks`
                         more to come in the future
            `verbose`: if True, print and graph the predicted results
            `show_history`: if True, graph the training history
        【Returns】
            `self.preds`: a list of dicts, each presented in the following form -
                                 {'window': prediction window,
                                 'data': the return values of function `split_data()`,
                                 'pred': the return values of function `predict_score()`,
                                 'model': prediction model,
                                 'history': training history}
        '''
        if ticker != self.ticker or ema != self.ema:
            self.get_features(ticker, ema=ema, log_form=log_form)
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
    
    
    def ensemble_prediction(self, ticker, ema_basket, dropout_basket, pred_params, verbose=True, show_history=False):
        '''
        ** Main Function **
        Get one ensemble prediction per window, with fine-tuned hyperparameters and stacked models.
        Stacking strategy: averaging up to 3 predictions with the best R-squared scores.

        【Parameters】
            `ema_basket`: a list of `ema` options used for fine-tuning.
            `dropout_basket`: a list of `dropout` options used for fine-tuning.
            `pred_params`: a dict containing other parameters for function `get_prediction()`.
        【Returns】
            `fine_tuning`: a dict of subdicts, each subdict corresponding to a model with specific window.
                           e.g. model_name == 'pred(50, 1)', meaning 50-day lookback, 1-day lookahead.
                            {model_name:{'r2': [r2_tuning_1, r2_tuning_2, ...],
                                        'preds': [y_pred_tuning_1, y_pred_tuning_2, ...],
                                        'newpreds': [future_pred_tuning_1, future_pred_tuning_2, ...],
                                        'stacked': [stacked_y_pred, stacked_future_pred] }}
                            For keys 'r2', 'preds' and 'newpreds', the length of their correpsonding values
                            should be (len(ema_basket)*len(dropout_basket)).
        '''
        self.fine_tuning = {}
        self.hypers = []
        for ema in ema_basket:
            for dropout in dropout_basket:
                self.hypers.append({'ema':ema, 'dropout':dropout})
                predicted = self.get_prediction(ticker, verbose=False,
                                                ema=ema, dropout=dropout, **pred_params)
                for i in range(len(predicted)):
                    model_name = '(%s,%s)'% (predicted[i]['window'][0], predicted[i]['window'][1])
                    if model_name not in self.fine_tuning:
                        self.fine_tuning[model_name] = {'r2':[], 'preds':[], 'newpreds':[], 'histories':[], 'models':[]}
                      
                    self.fine_tuning[model_name]['r2'].append(predicted[i]['pred'][2])
                    self.fine_tuning[model_name]['preds'].append(predicted[i]['pred'][-2])
                    self.fine_tuning[model_name]['newpreds'].append(predicted[i]['pred'][-1])
                    self.fine_tuning[model_name]['histories'].append(predicted[i]['history'])
                    self.fine_tuning[model_name]['models'].append(predicted[i]['model'])
        
        # stacking models
        keys = [k for k in self.fine_tuning.keys()]
        for i in range(len(keys)):
            model = self.fine_tuning[keys[i]]
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
        
        if verbose:
            self.allow_verbosity(fine_tuning=True)
        if show_history:
            self.show_history(fine_tuning=True)
        return self.fine_tuning

    
    def allow_verbosity(self, model_type='lstm', fine_tuning=False, show_period=120):
            
        fig, ax = plt.subplots(figsize=(16,5))
        y = []
        date = []
        future = []
        for i, pred in enumerate(self.preds):
            y_test = pred['data'][3]
            date_test = pred['data'][4]
#             last_date = datetime.strptime(date_test[-1],'%Y-%m-%d')
#             future_dates = [(last_date + timedelta(days=x+1)).strftime('%Y-%m-%d') for x in range(pred['window'][1])]
            future_dates = ['T + %s'% (x+1) for x in range(pred['window'][1])]
            date_test_fut = list(date_test[-show_period:]) + future_dates
            
            if self.log_form:
                y_test = np.exp(y_test)
            if len(y_test) > len(y):
                y = y_test
                date = date_test
            if len(future_dates) > len(future):
                future = future_dates

            print('='*15, '%s: Model (%s, %s)'% (self.ticker, pred['window'][0], pred['window'][1]),'='*15)
            print('Train Size:', pred['data'][0].shape, pred['data'][1].shape)
            print('Test Size:', pred['data'][2].shape, pred['data'][3].shape)
            print('Size of Data for Future Prediction:', pred['data'][-1].shape)
            
            
            if fine_tuning:     # when called by ensemble_prediction()
                keys = [k for k in self.fine_tuning.keys()]
                stacked_preds = self.fine_tuning[keys[i]]['stacked']
                
                r2_ensemble = r2_score(y_test, stacked_preds['preds'])
                
                print('Selected Hyperparameters:', stacked_preds['hypers'])
                print('R-Squared (ensemble): %.4f' % r2_ensemble)
                print('Future Prediction: %s' % [round(float(x),4) for x in stacked_preds['newpreds'][0]])
                
                pred_prices = [p[-1] for p in stacked_preds['preds']]
                pred_prices = pred_prices[-show_period:] + list(stacked_preds['newpreds'][0])
                ax.plot(date_test_fut, pred_prices, label='Ensemble '+keys[i])

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
        plt.ylabel(self.ticker)
        plt.legend()
        plt.show()
        
        
    def show_history(self, fine_tuning=False):
        if fine_tuning:
            keys = [k for k in self.fine_tuning.keys()]
            fig = plt.figure(figsize=(3*len(self.hypers), 2*len(keys)))
            j = 0
            for key in keys:
                for history in self.fine_tuning[key]['histories']:
                    ax = fig.add_subplot(len(keys), len(self.hypers), j+1)
                    fig.tight_layout()
                    ax.plot(history.history['loss'])
                    ax.plot(history.history['val_loss'])
                    if j < len(self.hypers):
                        plt.title('EMA: {}, Dropout: {}'.format(self.hypers[j]['ema'], self.hypers[j]['dropout']))
                    if j % len(self.hypers) == 0:
                        plt.ylabel('Model '+key)
                    if j >= len(self.hypers)*(len(keys)-1):
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