# -*- coding: utf-8 -*-

import scrapy

class MetricsItem(scrapy.Item):
    fticker = scrapy.Field()
    fname = scrapy.Field()
    ftype = scrapy.Field()
    date = scrapy.Field()
    stdev_1y = scrapy.Field()
    stdev_2y = scrapy.Field()
    stdev_3y = scrapy.Field()
    sharp_1y = scrapy.Field()
    sharp_2y = scrapy.Field()
    sharp_3y = scrapy.Field()
    current_nav = scrapy.Field()
    current_r_d = scrapy.Field()
    current_style = scrapy.Field()
    

class AllocationItem(scrapy.Item):
    fticker = scrapy.Field()
    current_stocks = scrapy.Field()
    current_bonds = scrapy.Field()
    current_cash = scrapy.Field()
    current_net_assets = scrapy.Field()
    last_net_assets = scrapy.Field()
    last2_net_assets = scrapy.Field()
    

class IndustriesItem(scrapy.Item):
    fticker = scrapy.Field()
    industry_1 = scrapy.Field()
    industry_1_pct = scrapy.Field()
    industry_2 = scrapy.Field()
    industry_2_pct = scrapy.Field()
    industry_3 = scrapy.Field()
    industry_3_pct = scrapy.Field()
    industry_4 = scrapy.Field()
    industry_4_pct = scrapy.Field()
    industry_5 = scrapy.Field()
    industry_5_pct = scrapy.Field()
    

# https://www.cnblogs.com/wsygdb/p/8316531.html
class ReturnItem(scrapy.Item):

    fticker = scrapy.Field()
    date = scrapy.Field()
    recent_1m = scrapy.Field() 
    recent_3m = scrapy.Field()
    recent_6m = scrapy.Field()
    recent_1y = scrapy.Field() 
    recent_2y = scrapy.Field() 
    recent_3y = scrapy.Field() 
#     serviceCharge = scrapy.Field()


class HistoryItem(scrapy.Item):
    fticker = scrapy.Field()
    hdate = scrapy.Field()
    nav = scrapy.Field()
    accu_nav = scrapy.Field()
    r_d = scrapy.Field()
    

class ManagerItem(scrapy.Item):
    fticker = scrapy.Field()
    ranking = scrapy.Field()
