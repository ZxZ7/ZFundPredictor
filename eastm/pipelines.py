# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html
import pymysql
from eastm.items import MetricsItem, AllocationItem, IndustriesItem, ReturnItem, HistoryItem, ManagerItem


class EastmPipeline(object):
    def __init__(self):
        self.conn = pymysql.connect(host='localhost',
                             user='root',
                             password='*****',   # your password
                             db='funds',
                             charset='utf8mb4')
        self.cursor = self.conn.cursor()
        self._sqlMetrics = None
        self._sqlAllocation = None
        self._sqlReturn = None
        self._sqlIndustries = None
        self._sqlHistory = None
        self._sqlManager = None
        
    
    def process_item(self, item, spider):
        if isinstance(item, MetricsItem):
            self.cursor.execute(self.sqlMetrics, (item['fticker'], item['fname'], item['ftype'], item['date'], item['stdev_1y'],
                                                  item['stdev_2y'], item['stdev_3y'], item['sharp_1y'],
                                                  item['sharp_2y'], item['sharp_3y'], item['current_nav'],
                                                  item['current_r_d'], item['current_style'],   # insert
                                                  item['fname'], item['ftype'], item['date'], item['stdev_1y'],
                                                  item['stdev_2y'], item['stdev_3y'], item['sharp_1y'],
                                                  item['sharp_2y'], item['sharp_3y'], item['current_nav'],
                                                  item['current_r_d'], item['current_style']))  # update

        elif isinstance(item, AllocationItem):
            self.cursor.execute(self.sqlAllocation, (item['current_stocks'], item['current_bonds'], item['current_cash'],
                                                     item['current_net_assets'], item['last_net_assets'], item['last2_net_assets'],
                                                     item['fticker'],                                                         # insert
                                                     item['current_stocks'], item['current_bonds'], item['current_cash'],
                                                     item['current_net_assets'], item['last_net_assets'], item['last2_net_assets'])) # update           
        elif isinstance(item, ReturnItem):
            self.cursor.execute(self.sqlReturn, (item['fticker'], item['date'], item['recent_1m'], item['recent_3m'], item['recent_6m'],
                                                     item['recent_1y'], item['recent_2y'], item['recent_3y'],                # insert
                                                     item['recent_1m'], item['recent_3m'], item['recent_6m'],
                                                     item['recent_1y'], item['recent_2y'], item['recent_3y']))             # update   
        elif isinstance(item, IndustriesItem):
            self.cursor.execute(self.sqlIndustries, (item['fticker'], item['industry_1'], item['industry_1_pct'], item['industry_2'],
                                                     item['industry_2_pct'], item['industry_3'], item['industry_3_pct'], 
                                                     item['industry_4'], item['industry_4_pct'], item['industry_5'], item['industry_5_pct'],
                                                     item['industry_1'], item['industry_1_pct'], item['industry_2'],
                                                     item['industry_2_pct'], item['industry_3'], item['industry_3_pct'], 
                                                     item['industry_4'], item['industry_4_pct'], item['industry_5'], item['industry_5_pct']))
        elif isinstance(item, HistoryItem):
            self.cursor.execute(self.sqlHistory, (item['fticker'], item['hdate'], item['nav'], item['accu_nav'], item['r_d']))
        
        elif isinstance(item, ManagerItem):
            self.cursor.execute(self.sqlManager, (item['fticker'], item['ranking'], item['ranking']))
        
        self.conn.commit()
        return item
    

    
    @property
    def sqlMetrics(self):
        if not self._sqlMetrics:
            self._sqlMetrics = '''
            insert into all_funds(id, fticker, fname, ftype, date, stdev_1y, stdev_2y, stdev_3y, sharp_1y, sharp_2y, sharp_3y, current_nav, current_r_d, current_style)
            values(null, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            on duplicate key update fname=%s, ftype=%s, date=%s, stdev_1y=%s, stdev_2y=%s, stdev_3y=%s, sharp_1y=%s, sharp_2y=%s, sharp_3y=%s, current_nav=%s, current_r_d=%s, current_style=%s;
            '''
            return self._sqlMetrics
        return self._sqlMetrics
    
    
    @property
    def sqlAllocation(self):
        if not self._sqlAllocation:
            self._sqlAllocation = '''

            insert into all_funds(id, current_stocks, current_bonds, current_cash, current_net_assets, last_net_assets, last2_net_assets, fticker)
            values(null, %s, %s, %s, %s, %s, %s, %s)
            on duplicate key update current_stocks=%s, current_bonds=%s, current_cash=%s, current_net_assets=%s, last_net_assets=%s, last2_net_assets=%s;
            '''
            return self._sqlAllocation
        return self._sqlAllocation
    
    
    @property
    def sqlReturn(self):
        if not self._sqlReturn:
            self._sqlReturn = '''

            insert into all_funds(id, fticker, date, recent_1m, recent_3m, recent_6m, recent_1y, recent_2y, recent_3y)
            values(null, %s, %s, %s, %s, %s, %s, %s, %s)
            on duplicate key update recent_1m=%s, recent_3m=%s, recent_6m=%s, recent_1y=%s, recent_2y=%s, recent_3y=%s;
            '''
            return self._sqlReturn
        return self._sqlReturn

    
    @property
    def sqlIndustries(self):
        if not self._sqlIndustries:
            self._sqlIndustries = '''
            
            insert into all_funds(id, fticker, industry_1, industry_1_pct, industry_2, industry_2_pct, industry_3, industry_3_pct, industry_4, industry_4_pct, industry_5, industry_5_pct)
            values(null, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            on duplicate key update industry_1=%s, industry_1_pct=%s, industry_2=%s, industry_2_pct=%s, industry_3=%s, industry_3_pct=%s, industry_4=%s, industry_4_pct=%s, industry_5=%s, industry_5_pct=%s;
            '''
            return self._sqlIndustries
        return self._sqlIndustries


#             update all_funds
#             set current_stocks=%s, current_bonds=%s, current_cash=%s, current_net_assets=%s, last_net_assets=%s, last2_net_assets=%s
#             where fticker=%s;

    @property
    def sqlHistory(self):
        if not self._sqlHistory:
            self._sqlHistory = '''

            insert into histories(fticker, hdate, nav, accu_nav, r_d)
            values(%s, %s, %s, %s, %s)
            on duplicate key update fticker=fticker;
            '''
            return self._sqlHistory
        return self._sqlHistory
    
    
    @property
    def sqlManager(self):
        if not self._sqlManager:
            self._sqlManager = '''

            insert into all_funds(id, fticker, manager_ranking)
            values(null, %s, %s)
            on duplicate key update manager_ranking=%s;
            '''
            return self._sqlManager
        return self._sqlManager
