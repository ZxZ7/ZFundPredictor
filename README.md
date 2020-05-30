# ZFundPredictor (ver. 1.2.0)


This project aims to build a stable price predictor for open-end mutual funds available in China's fund market.

It consists of:

* A **web-scraping** framework that extracts data of funds from http://fund.eastmoney.com/, and stores the data in a local database.
* An **ETL** pipeline that mainly transforms the raw data into datasets for machine learning models.
* A machine learning/deep learning **prediction tool** (currently based on LSTM only - more to come in the future).


<br>

## Prerequisites

This project is written in `Python` with data stored into a `MySQL` database.

Key libraries include `Scrapy` · `PyMySQL` · `Numpy` · `Pandas` · `Matplotlib` · `Seaborn` · `SciKit-Learn` · `TensorFlow (tf.keras)`

<br>

## Getting Started

* ### Scrapy Framework

  The following raw data of a fund will be collected using spider `em.py`:
  
    - Fund Name, Ticker, Type (stock/stock index/bond/hybrid)
    - Current Investment Style (large-/mid-/small-cap, value/growth)
    - Current Net Assets, Net Assets of the last two reporting periods
    - Current Asset Allocation (% of stock/bond/cash)
    - Current Industry Allocation (top 5, and their respective weights)
    - Manager Performance: Current Ranking Score in similar funds (between 0 and 1, the smaller the better)
    - Historical Prices from launching
    - Performance Metrics: Fund Returns (recent 1d/1m/6m/1y/2y/3y), Standard Deviation (recent 1y/2y/3y), Sharpe Ratio (recent 1y/2y/3y)

  Note: data are updated daily by the scraped website, except for asset size, asset allocation and industry allocation, which are updated per reporting period (season).
  
  Complete source code of this framework can be found [here](https://github.com/ZxZ7/ZFundPredictor/tree/master/eastm).


* ### Data Prepossessing

  ```
  from ZFundETL import FundETL
  ETL = FundETL()
  funds, categorical = ETL.quick_prepossessing()
  ```
  
  For more details, please check the [demo](https://github.com/ZxZ7/ZFundPredictor/tree/master/demo).

  This process mainly generates two datasets.

  - `funds`: contains the **historical daily prices** of available funds during the selected period, plus the **daily returns** of the benchmarks (stock or bond index).

  - `categorical`: contains all the short-term invariant features, including fund types, fund styles, asset size, ranking scores (manager performance), asset allocation, and industry allocation.



* ### Predictor

  ```
  from ZFundPredictor import FundPredictor
  predictor = FundPredictor(funds)
  single_prediction = predictor.get_prediction(ticker, **params)
  ensemble_prediction = predictor.ensemble_prediction(ticker, **tune_params)
  ```
  
  For more details, please see the following section and the [demo](https://github.com/ZxZ7/ZFundPredictor/tree/master/demo).


<br>


## Predicting Strategy

The current predicting algorithm is based on LSTM with sliding windows.


* ### Key Terms

    - **Lookback**: the length of a sliding window, indicating how many periods we are looking back to the past.
    - **Lookahead**: the number of periods that we want to predict into the future.
    - **MA**: moving average indicators.
      - `dist_EMA`: the distances between fund price and x-day exponential moving average (EMA) of the fund on each day.
      - `signal_EMA`: one-time EMA signal. `-1` if the shorter EMA crosses above the longer EMA on that day, `1` vice versa, and `0` if there is no cross.
      - `status_BB`: short-term status of the Bollinger Bands indicator. `-1` if the value of x-day BB indicator is below the lower band, `1` if it is above the upper band, and `0` otherwise.
  

  Currently in use:<br>
    - **`lookback, lookahead`:** [50, 1], [120, 2] and [120, 5]
    - **MA:** [None], [20, 50] or [5, 50]
      - `'ema'` - adopt only the EMA indictors `dist_ema` (e.g. `dist_ema5` refers to 5-day EMA, and `dist_ema50` refers to 50-day EMA).
      - `'all'` - adopt all available indicators.


  
* ### Splitting Method
  
  <img src="https://github.com/ZxZ7/ZFundPredictor/blob/master/splitting_strategy.png" width="700">
 
 
* ### LSTM Time Series
  #### Architecture and training/predicting methods
    
    A model with a 120-day Lookback period and a 5-day Lookahead period would look like this:
    
    | Layer (type)            |   Output Shape            |  Param #  |
    | ------------- |:-------------:| -----:|
    |lstm_0 (LSTM)       |           (None, 120, 256)      |    268288    |
    |dropout_0 (Dropout)    |        (None, 120, 256)      |    0         |
    |lstm_1 (LSTM)          |      (None, 256)        |       525312    |
    |dropout_1 (Dropout)    |      (None, 256)         |      0         |
    |dense_0 (Dense)           |     (None, 32)          |      8224      |
    |dense_1 (Dense)         |     (None, 5)            |     165       |
    |||Total params: 801,989|
    |||Trainable params: 801,989|
    
  
    When `get_prediction()` is called, the training set will go through the above layers and a 	***single*** model prediction will be generated with the provided parameters.
    
     `ensemble_prediction()` allows a basket of **MA** options and a basket of **dropout** options. Each combination will form a new model, and based on their **R-squared** scores, the final prediction will be the weighted average of some of these models.
     
     An illustration of `ensemble_prediction()`:
     
     <img src="https://github.com/ZxZ7/ZFundPredictor/blob/master/LSTM_ensemble.png" width="700">
     
     
  #### Hyperparameter Tuning
  
    - **Epochs**: set to a large number
    - **Batch Size**: 300 or 400
    - **Learning Rate**: 0.0005
    - **Early Stopping**: `monitor='val_loss', min_delta=1e-5, patience=5, restore_best_weights=True`
    On average, the training stops between epoch 10 and 20.
    - **Dropout**: 0.2 or 0.3
    
<br>

## Future plan

  - The future version will introduce categorical features after the LSTM time series forcasting process to account for individual effects and to provide more stable predictions. 
  - So far, the models have only been trained and tested on funds that mainly invest in stocks. In the future, this predictor might be generalized to include bond funds as well.
  - A fund portfolio builder and monitor...

<br>

## Meta

Ideas and contributions welcome.

Contact: zhexin-zhang@hotmail.com

Distributed under the [MIT](https://choosealicense.com/licenses/mit/) license.
