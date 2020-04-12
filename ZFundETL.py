import math
from datetime import datetime, timedelta, date
import re
import pymysql
import yfinance as yf

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns



class FundETL:
    def __init__(self):
        self.industries = ['制造业', '金融业', '信息传输、软件和信息技术服务业', '房地产业',
              '交通运输、仓储和邮政业', '农、林、牧、渔业', '批发和零售业',
              '采矿业', '住宿和餐饮业', '租赁和商务服务业', '水利、环境和公共设施管理业',
              '文化、体育和娱乐业', '科学研究和技术服务业', '卫生和社会工作', '建筑业',
              '电力、热力、燃气及水生产和供应业', '教育', '综合']

        sns.set_style("darkgrid")
        sns.set_context("notebook")


    def sql_queries(self):
        '''
        Getting data from MySQL.
        '''
        connection = pymysql.connect(host='localhost',
                                     user='root',
                                     password='*****',
                                     db='funds',
                                     charset='utf8mb4')

        cursor = connection.cursor()

        sql = '''
        CREATE OR REPLACE VIEW picked AS
        SELECT a.* FROM (
            SELECT fticker FROM histories
            WHERE hdate >= '2015-01-05'
            AND r_d != 0
            GROUP BY fticker
            HAVING COUNT(*) > CEIL(DATEDIFF(CURDATE(), '2016-01-05')*0.75)) f
        JOIN (
            SELECT * FROM all_funds
            WHERE current_stocks >= 75
            AND current_net_assets > 2) a
        ON a.fticker = f.fticker
        WHERE current_bonds < 25 OR current_bonds IS NULL;
        '''
        cursor.execute(sql)
        connection.commit()


        sql = '''
        SELECT h.fticker, h.hdate, accu_nav FROM histories h
        RIGHT JOIN picked p
        ON h.fticker = p.fticker
        WHERE h.hdate >= '2015-01-05';
        '''

        self.funds_sql = pd.read_sql(sql, connection)


        sql = '''
        SELECT
            fticker, ftype, current_style, current_net_assets,
            current_stocks, current_bonds, current_cash,
            industry_1, industry_1_pct, industry_2, industry_2_pct, industry_3, industry_3_pct,
            industry_4, industry_4_pct, industry_5, industry_5_pct, manager_ranking
        FROM picked;
        '''
        # Short term invariant variables
        self.st_invariants = pd.read_sql(sql, connection)

        connection.close()
        cursor.close()



    def get_index(self, index_type='stock'):
        '''
        Import data of index.
        '''
        if index_type == 'stock':
            sindex = yf.download("000001.ss", start=str(self.funds_sql['hdate'].min()),
                                 end=str(self.funds_sql['hdate'].max()+timedelta(days=1)))
            # update missing data
            sindex.loc['2019-12-19'] = [3017.15,3021.42,3007.99,3017.07,3017.07,208600]
            sindex.loc['2019-04-29'] = [3090.63,3107.76,3050.03,3062.50,3062.50,292100]
            sindex.loc['2019-04-30'] = [3052.62,3088.41,3052.62,3078.34,3078.34,222300]
            sindex['sindex_r'] = (sindex['Adj Close'] - sindex['Adj Close'].shift(1)) / sindex['Adj Close'].shift(1)*100
            sindex['Date'] = sindex.index
            sindex = sindex.set_index(pd.to_datetime(sindex['Date']).dt.date).drop(columns='Date').sort_index()
            return sindex

        if index_type == 'bond':
            tbond = pd.read_csv(r'China 10-Year Bond Yield Historical Data.csv')
            tbond = tbond.set_index(pd.to_datetime(tbond['Date']).dt.date).sort_index()
            tbond['tbond_d'] = tbond['Change %'].str.rstrip('%').astype('float') / 100.0
            tbond.drop(columns=['Date', 'Change %'], inplace=True)
            return tbond


    def find_missing_values(self, show_results=True):
        '''
        Find the funds with missing prices and returns the their tickers.
        '''
        drop_tickers = self.funds_sql[self.funds_sql['accu_nav'].isnull()]['fticker'].unique()
        
        if show_results:
            n_col = math.ceil(len(drop_tickers)/2)

            fig = plt.figure(figsize=(2*n_col,4))
            fig.suptitle('Funds With Missing Values in Historical Prices', fontsize=14)
            # color = next(ax._get_lines.prop_cycler)['color']
            for i, ticker in enumerate(drop_tickers):
                _null = self.funds_sql[self.funds_sql['fticker'] == ticker]['accu_nav'].isnull().sum()
                _notnull = self.funds_sql[self.funds_sql['fticker'] == ticker]['accu_nav'].notnull().sum()

                ax = fig.add_subplot(2, n_col, i+1)
                fig.tight_layout()        
                ax.pie([_null, _notnull], radius=1.1, wedgeprops=dict(width=0.2),
                       colors=sns.color_palette('twilight_shifted', n_colors=2),
                       autopct=lambda pct: '{:.2f}%\n(# NA: {:.0f})'.format(pct, _null) 
                                if int(pct) == int(_null/(_notnull+_null)*100) else '')
                plt.xlabel(ticker)

            print('Number of funds with missing values in *Historical Prices*:', len(drop_tickers))
            drop_tickers2 = self.st_invariants[(self.st_invariants['industry_1'].isnull()) | 
                                         (self.st_invariants['manager_ranking'].isnull())]['fticker']
            print('Number of funds with missing values in *Short-Term Invariant Variables*:', len(drop_tickers2))
            drop_tickers = set(drop_tickers) | set(drop_tickers2)
            print('Total number of funds to be dropped because of missing data:', len(drop_tickers))
            plt.show()
        
        return drop_tickers


    def count_days(self, show_results=True):
        '''
        Count trading days of the funds and return the tickers of funds with the most common length.
        '''
        funds_length = self.funds_sql.groupby('fticker')['hdate'].count()
        count_per_length = funds_length.groupby(funds_length.values).count()
        
        max_count = count_per_length.max()
        rest = count_per_length.sum() - max_count
        most_common_length = count_per_length[count_per_length == max_count].index[0]
        
        tickers_common_length = funds_length[funds_length == most_common_length].index
        
        if show_results:
            fig, ax = plt.subplots(figsize=(3,3))
            ax.pie([max_count, rest], wedgeprops=dict(width=0.15), radius=0.9,
                   colors=sns.color_palette('twilight_shifted', n_colors=2),
                   autopct=lambda pct: '{:.2f}%\n(# funds: {})'.format(pct, max_count) if pct>50 else '')
            plt.title('Available Length of Funds', fontsize=14)
            plt.legend([str(most_common_length)+' days', 'Other lengths'], loc='lower center', ncol=2)
            plt.show()
        return tickers_common_length


    def ticker_filter(self, show_results=True):
        '''
        Filter out the funds with missing values.
        '''
        drop_tickers = self.find_missing_values(show_results)
        tickers_common_length = self.count_days(show_results)
        return np.array([t for t in tickers_common_length if t not in drop_tickers])



    def get_funds(self, selected_tickers, stock_index, bond_index=None, show_results=True):
        '''
        Build the `funds` dataset.
        '''
        for ticker, histories in self.funds_sql.groupby('fticker'):
            if ticker in selected_tickers:
                if ticker == selected_tickers[0]:
                    funds = pd.DataFrame(index=histories['hdate'])
                funds[ticker] = histories['accu_nav'].values
            
        fund_std = funds.apply(lambda x: x.std())
        cutoff = 0.75
        highly_volatile = fund_std[fund_std > cutoff]
        
        if show_results:
            fig, ax = plt.subplots(figsize=(10, 1.5))
            sns.boxplot(data=fund_std, orient='h', color='mediumslateblue', width=0.3, ax=ax)
            ax.vlines(cutoff, -0.5, 0.5, linestyles='dashed', colors='orange')

            for order, sorting_idx in enumerate(highly_volatile.argsort()[::-1]):
                stv = highly_volatile[sorting_idx]
                stv_ticker = fund_std[fund_std == stv].index[0]
                arrowprops = {'arrowstyle':'simple,head_length=0.8,head_width=0.6,tail_width=0.3',
                              'ec':None, 'facecolor':'orange', 'connectionstyle':'arc3',
                              'shrinkA':0, 'shrinkB':5}
                if order%4 == 0:
                    ax.text(stv-0.06, -0.1, stv_ticker)
                elif order%4 == 2:
                    plt.annotate(stv_ticker, xy=(stv, 0), xytext=(stv-0.01, -0.3), arrowprops=arrowprops)
                elif order%4 == 1:
                    ax.text(stv-0.06, 0.2, stv_ticker)
                else:
                    plt.annotate(stv_ticker, xy=(stv, 0), xytext=(stv-0.04, 0.4), arrowprops=arrowprops)
            plt.yticks([0], ['STDEV'])
            plt.title('Volatility of Funds', fontsize=14)
            plt.show()
        

        new_cols = [c for c in funds.columns if c not in highly_volatile.index]
        funds = funds.loc[:, new_cols]
        
        if bond_index:
            funds_ = pd.concat([funds, stock_index, bond_index], axis=1, join='inner').dropna()
        else:
            funds_ = pd.concat([funds, stock_index], axis=1, join='inner').dropna()
        funds_.index.rename('Date', inplace=True)

        
        if show_results:
            print('Removing funds with excessive volatility:', [c for c in highly_volatile.index])
            print('Dates further dropped:', [str(i) for i in funds.index if i not in funds_.index])
            print('Final available funds:', funds.shape[1])
            print('Final available days:', funds_.shape[0])
            
        self.funds = funds_

        return self.funds


    def build_categories(self):
        '''
        Build the `categorical` dataset.
        '''
        categorical = pd.DataFrame(index=self.st_invariants['fticker'])
        
        # label categories
        categorical['fund_type'] = self.st_invariants['ftype'].astype('category').values
        categorical['fund_style'] = self.st_invariants['current_style'].astype('category').values
        
        # numerical to categorical
        categorical['asset_size'] = pd.qcut(self.st_invariants['current_net_assets'].values, 4)
        categorical['ranking_score'] = pd.cut(self.st_invariants['manager_ranking'].values, [0,0.25,0.5,0.75,1])
        
        for col in ['current_stocks', 'current_bonds', 'current_cash']:
            categorical[col] = (self.st_invariants[col]/100).fillna(0).values
        
        # one-hot encoding for industries 
        weighted_oh = []
        for x in range(1,6):
            _oh = pd.get_dummies(self.st_invariants['industry_%s'%x].values)
            for ind in range(_oh.shape[1]):
                _oh.iloc[:,ind] = _oh.iloc[:,ind]*self.st_invariants['industry_%s_pct'%x].values/100
            weighted_oh.append(_oh)
        
        industry_w = pd.DataFrame(index=self.st_invariants['fticker'], columns=self.industries).applymap(lambda x: 0)
        
        columns = []
        for num, indust in enumerate(self.industries):
            for x in range(1,6):
                if indust in set(self.st_invariants['industry_%s'%x]):
                    industry_w[indust] = industry_w[indust].values + weighted_oh[x-1][indust].values
            columns.append('ind_%s'%num) 
        industry_w.columns = columns
        
        # conbine all
        categorical = pd.concat([categorical, industry_w], axis=1)
        tickers_ = [t for t in self.funds.columns if t not in ['sindex_r', 'tbond_d']]
        
        self.categorical = categorical.loc[tickers_]

        return self.categorical


    def categorical_summary(self):
        '''
        Summarize and graph the `categorical` dataset.
        '''
        industry_count = len(self.industries)
        dicts = [{'混合型':'hybrid', '股票型':'stock', '股票指数':'stock index'},
                 {'大盘价值':'large value', '大盘平衡':'large balanced', '中盘成长':'large growth',
                  '中盘价值':'mid value', '中盘平衡':'mid balanced', '中盘成长':'mid growth',
                  '小盘价值':'small value', '小盘平衡':'small balanced', '小盘成长':'small growth'}]
        groups = ['fund_type', 'fund_style', 'asset_size', 'ranking_score']
        industry_w = self.categorical.iloc[:,-industry_count:]
        allocation = self.categorical.loc[:,['current_stocks','current_bonds','current_cash']]
        
        fig = plt.figure(figsize=(14,3.5))
        cmap = plt.get_cmap('tab20b')
        fig.suptitle('Categorical Features', fontsize=16)
        
        for i,feat in enumerate(groups):
            ax = fig.add_subplot(1,4,i+1)
    #         fig.subplots_adjust(wspace=None, hspace=None, top=1.2)
            fig.tight_layout()
            grouped = self.categorical[groups].groupby(feat)[feat]
            ax.pie(grouped.count(), radius=0.85, wedgeprops=dict(width=0.15),
                   colors=sns.color_palette('twilight_shifted', n_colors=len(grouped)),
    #                colors=cmap(np.arange(len(grouped))*5),
                   autopct=lambda pct: 
                           '{:.2f}%\n({:.0f})'.format(pct, self.categorical.shape[0]*pct/100)
                           if pct>5 else '')
            legend_param = {'loc':'lower center', 'ncol':2}
            if i in [0, 1]:
                plt.legend([dicts[i][idx] for idx, group in grouped], **legend_param)
            else:
                plt.legend([idx for idx, group in grouped], **legend_param)
            plt.title(re.sub('_',' ', feat).capitalize(), fontsize=13, pad=-20)
        
        fig, ax1 = plt.subplots(figsize=(12,4))
        sns.boxplot(data=pd.concat((allocation, industry_w), axis=1), ax=ax1, width=0.4,
                    palette=sns.color_palette('Set2', n_colors=len(industry_w.columns)+3))
        plt.xticks(range(len(industry_w.columns)+3),
                   ['Stocks', 'Bonds', 'Cash']+['Industry '+col.strip('ind_') for col in industry_w.columns], rotation=90)
        plt.ylabel('Weights')
        plt.show()
        
        industry_dict = {'住宿和餐饮业':'Hospitality & Catering',
         '租赁和商务服务业':'Lease & Business Services',
         '水利、环境和公共设施管理业':'Water Conservancy, Environment & Public Facilities Management',
         '金融业':'Finance',
         '文化、体育和娱乐业':'Culture, Sports & Entertainment',
         '房地产业':'Real Estate',
         '科学研究和技术服务业':'Scientific Research & Technical Services',
         '交通运输、仓储和邮政业':'Transportation, Warehousing & Postal Services',
         '批发和零售业':'Wholesale & Retail Trade',
         '卫生和社会工作':'Health & Social Work',
         '农、林、牧、渔业':'Agriculture, Forestry, Animal Husbandry & Fishery',
         '综合':'Comprehensive',
         '电力、热力、燃气及水生产和供应业':'Power, Heat, Gas & Water Production and Supply',
         '建筑业':'Construction',
         '制造业':'Manufacturing',
         '采矿业':'Mining',
         '信息传输、软件和信息技术服务业':'Information Transmission, Software & Information Technology Services',
         '教育':'Education'}
        
        industries_ = self.industries + [industry_dict[ind] for ind in self.industries]
        industries_ = pd.DataFrame(np.array(industries_).reshape(2,-1),
                                   columns=industry_w.columns, index=['行业','Industry'])
        summary = industry_w.describe()[1:].applymap(lambda x: round(x,4))
        summary = pd.concat((industries_, summary),axis=0)
        return summary


    def quick_prepossessing(self):
        '''
        Generate the `funds` and `categorical` datasets without showing the process.
        '''
        self.sql_queries()
        sindex = self.get_index()
        selected_tickers = self.ticker_filter(show_results=False)
        funds = self.get_funds(selected_tickers, sindex['sindex_r'], show_results=False)
        categorical = self.build_categories()

        return funds, categorical


    def save_files(self, path, date):
        self.funds.to_csv(path+f'funds_{date}.csv')
        self.categorical.to_csv(path+f'categorical_{date}.csv', index=False)


    def read_files(self, path, date):
        funds = pd.read_csv(path+f'funds_{date}.csv')
        funds = funds.set_index(pd.to_datetime(funds['Date']).dt.date).drop(columns='Date').sort_index()
        categorical = pd.read_csv(path+f'categorical_{date}.csv')
        categorical['fticker'] = [t for t in funds.columns if t not in ['sindex_r', 'tbond_d']]
        categorical.set_index('fticker', inplace=True)

        self.funds = funds
        self.categorical = categorical

        return funds, categorical