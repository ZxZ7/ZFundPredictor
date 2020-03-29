# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
import requests
import re
import pandas as pd
from eastm.items import MetricsItem, AllocationItem, IndustriesItem, ReturnItem, HistoryItem, ManagerItem
import json



class EmSpider(CrawlSpider):
    name = 'em'
    allowed_domains = ['eastmoney.com']


    def __init__(self):
        basics_resp = requests.get('http://fund.eastmoney.com/js/fundcode_search.js')
        basics = basics_resp.content.decode('utf-8')
        basics = re.sub(r'\ufeffvar r = \[', '', basics)
        basics = re.sub(r'];', '', basics)
        basics = re.sub(r'"', '', basics)
        basics = re.findall('\[(.*?)\]', basics)

        funds = []
        for idx in range(len(basics)):
            fund = [i for i in [basics[idx].split(',')[0]]+basics[idx].split(',')[2:4] ]
            if '(后端)' not in fund[1] and fund[2] in ['混合型', '股票指数', '股票型']:
                funds.append(fund)
        
        self.funds_df = pd.DataFrame(funds, columns=['fticker', 'fname', 'ftype'])            

            

    def start_requests(self):
        
        performance_url = 'https://fundapi.eastmoney.com/fundtradenew.aspx?ft=pg&sc=1n&st=desc&pi=1&pn=5000&cp=&ct=&cd=&ms=&fr=&plevel=&fst=&ftype=&fr1=&fl=0&isab='
        yield scrapy.Request(performance_url, callback=self.parse_performance)    ###
        

        ## Data of industry allocation and asset allocation are updated by season
        ## if crawling for the first time, please allow all links

        for ti in self.funds_df['fticker']:
            metrics_url = f'http://fundf10.eastmoney.com/tsdata_{ti}.html'
#             industries_url = f'http://fundf10.eastmoney.com/hytz_{ti}.html'
#             allocation_url = f'http://fundf10.eastmoney.com/zcpz_{ti}.html'
            histories_url = f'http://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code={ti}&page=1&per=49'
#             managers_url = f'http://fundf10.eastmoney.com/jjjl_{ti}.html'
    
            
            yield scrapy.Request(metrics_url, callback=self.parse_metrics)     ###
#             yield scrapy.Request(industries_url, callback=self.parse_industries)
#             yield scrapy.Request(allocation_url, callback=self.parse_allocation)
            yield scrapy.Request(histories_url, callback=self.parse_histories)
#             yield scrapy.Request(managers_url, callback=self.parse_managers)

               
    def parse_metrics(self, response):
        
        check_open = response.xpath("//p[@class='row']//text()")
        if '暂停' in check_open[2].get().strip() + check_open[6].get().strip():
            return
        
        f = response.xpath("//h4/a/text()").get().strip()
        fname = f.split(r' (')[0].strip()
        fticker = re.sub(r'\)', '', f.split(' ')[1])
        fticker = re.sub(r'\(', '', fticker)

        ftype = self.funds_df[self.funds_df['fticker'] == fticker]['ftype'].values[0]
        
        
        date = response.xpath("//p[@class='row row1']/label[2]/text()")[0].get().strip()
        date = '2020-'+re.findall(r'\d\d-\d\d', date)[0]
        

        current_p_r = response.xpath("//p//b[@class]/text()")[0].get().strip()
        current_p_r = re.sub(' \)', '', current_p_r)
        
        current_nav = current_p_r.split(" ( ")[0]
        try:
            current_nav = float(current_nav)
        except ValueError:
            current_nav = None

        current_r_d = current_p_r.split(" ( ")[1].strip('%')
        try:
            current_r_d = float(current_nav)
        except ValueError:
            current_r_d = None               

        
        nums = response.xpath("//td[@class='num']/text()")
        num_list = []
        for i in range(6):
            nm = nums[i].get().strip().strip('%')
            if nm == '--':
                nm = None
            else:
                nm = float(nm)
            num_list.append(nm)
        stdev_1y, stdev_2y, stdev_3y, sharp_1y, sharp_2y, sharp_3y = num_list
        
        
        style = response.xpath("//table[@class='fgtb']//td/text()")
        current_style = style[1].get().strip()
        # style_19q3 = style[3].get().strip()
        # style_19q2 = style[5].get().strip()
        # style_19q1 = style[7].get().strip()
    
        item = MetricsItem(fname=fname, ftype=ftype, fticker=fticker, date=date,
                   current_nav=current_nav, current_r_d=current_r_d,
                   stdev_1y=stdev_1y, stdev_2y=stdev_2y, stdev_3y=stdev_3y,
                   sharp_1y=sharp_1y, sharp_2y=sharp_2y, sharp_3y=sharp_3y,
                   current_style=current_style)

        print('='*30)
        yield item


    
    def parse_allocation(self, response):
        
        check_open = response.xpath("//p[@class='row']//text()")
        if '暂停' in check_open[2].get().strip() + check_open[6].get().strip():
            return
        
        f = response.xpath("//h4/a/text()").get().strip()
        # fname = f.split(r' (')[0].strip()
        fticker = re.sub(r'\)', '', f.split(' ')[1])
        fticker = re.sub(r'\(', '', fticker)        
        
        allocation = response.xpath("//td[@class='tor']/text()")
        allo_list = []
        for a in [0,1,2,3,7,11]:
            allo = allocation[a].get().strip().strip('%')
            if allo == '---':
                allo = None
            else:
                allo = float(allo)
            allo_list.append(allo)
        current_stocks, current_bonds, current_cash, current_net_assets, last_net_assets, last2_net_assets = allo_list

        item = AllocationItem(fticker=fticker, current_stocks=current_stocks, current_bonds=current_bonds,
                              current_cash=current_cash, current_net_assets=current_net_assets,
                              last_net_assets=last_net_assets, last2_net_assets=last2_net_assets)        

        print('='*30)
        yield item
        

        
    def parse_industries(self, response):
        
        check_open = response.xpath("//p[@class='row']//text()")
        if '暂停' in check_open[2].get().strip() + check_open[6].get().strip():
            return
        
        f = response.xpath("//h4/a/text()").get().strip()
        # fname = f.split(r' (')[0].strip()
        fticker = re.sub(r'\)', '', f.split(' ')[1])
        fticker = re.sub(r'\(', '', fticker)
        
        industries = response.xpath("//div[@class='box'][1]//td/text()")
        indust_list = []
        
        if len(industries) < 21:
            indust_idx = [1,2,6,7,11,12,16,17]
        else:
            indust_idx = [1,2,6,7,11,12,16,17,21,22]
        for ind in indust_idx:
            indust = industries[ind].get().strip().strip('%')
            if indust == '--' or indust == '---':
                indust = None
            elif ind in [2,7,12,17,22]:
                indust = float(indust)
            indust_list.append(indust)
            
        industry_1, industry_1_pct, industry_2, industry_2_pct, industry_3, industry_3_pct, industry_4, industry_4_pct, industry_5, industry_5_pct = indust_list
        
        item = IndustriesItem(fticker=fticker, industry_1=industry_1, industry_1_pct=industry_1_pct, industry_2=industry_2,
                              industry_2_pct=industry_2_pct, industry_3=industry_3, industry_3_pct=industry_3_pct,
                              industry_4=industry_4, industry_4_pct=industry_4_pct, industry_5=industry_5, industry_5_pct=industry_5_pct)
        
        print('='*30)
        yield item

        
        
    def parse_performance(self,response):
        datas = response.body.decode('UTF-8')

        # get json codes
        datas = datas[datas.find('{'):datas.find('}')+1] # from the first ``{``, end with ``}``

        datas = datas.replace('datas', '\"datas\"')
        datas = datas.replace('allRecords', '\"allRecords\"')
        datas = datas.replace('pageIndex', '\"pageIndex\"')
        datas = datas.replace('pageNum', '\"pageNum\"')
        datas = datas.replace('allPages', '\"allPages\"')

        jsonBody = json.loads(datas)
        jsonDatas = jsonBody['datas']

        for data in jsonDatas:
            fundsArray = data.split('|')
            if fundsArray[0] in self.funds_df['fticker'].values:
                fticker = fundsArray[0]
                date = fundsArray[3]
                 
                fund_list = []
                for r in range(7,13):

                    if fundsArray[r] == '':
                        fundsArray[r] = None
                    else:
                        fundsArray[r] = float(fundsArray[r])
                    fund_list.append(fundsArray[r])
                                    
                recent_1m, recent_3m, recent_6m, recent_1y, recent_2y, recent_3y = fund_list              

                item = ReturnItem(fticker=fticker, date=date, recent_1m=recent_1m, recent_3m=recent_3m, recent_6m=recent_6m,
                                  recent_1y=recent_1y, recent_2y=recent_2y, recent_3y=recent_3y)
                print('='*30)
                yield item
                

    def parse_histories(self,response):
        
        fticker = re.findall('code=(.+?)&',response.url)[0]
        curpage = int(re.findall('page=(.+?)&',response.url)[0])
                
        h_body = response.body.decode('UTF-8')
        
        if curpage == 1:   
            if '暂停' in h_body:
                return       
        
        ## ***To parse all historical data***
        ## if crawling for the first time, please uncomment the 4 lines below

#             pages = int(re.findall(',pages:(.*),', h_body)[0])
#             for p in range(2, pages):
#                 next_url = f'http://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code={fticker}&page={p}&per=49'
#                 yield scrapy.Request(next_url, callback=self.parse_histories)
        
        histories = h_body[h_body.find('{'):h_body.find('}')+1]

        histories = histories.replace(' content', '\"content\"')
        histories = histories.replace('records', '\"records\"')
        histories = histories.replace('pages', '\"pages\"')
        histories = histories.replace('curpage', '\"curpage\"')

        jsonBody = json.loads(histories)
        jsonDatas = jsonBody['content']
        jsonDatas = [i for i in re.findall('>(.*?)<',jsonDatas)][21:]
        jsonDatas = [[jsonDatas[i],jsonDatas[i+2],jsonDatas[i+4],jsonDatas[i+6]] for i in range(len(jsonDatas)) if i%16 == 0]
                
        for history in jsonDatas:
            hdate, nav, accu_nav, r_d = history
            r_d = r_d.strip('%')
            try:
                nav = float(nav)
            except ValueError:
                nav = None
            try:
                accu_nav = float(accu_nav)
            except ValueError:
                accu_nav = None
            try:
                r_d = float(r_d)
            except ValueError:
                if nav != None:
                    r_d = 0.0
                else:
                    r_d = None
            item = HistoryItem(fticker=fticker, hdate=hdate, nav=nav, accu_nav=accu_nav, r_d=r_d)
            print('='*30)
            yield item
        
        
    def parse_managers(self,response):
        
        fticker = re.findall('jjjl_(.+)\.', response.url)[0]
        ranks = response.xpath('//tr[@style="background:#D2E2FF;"]/td[@class="tor"]/text()').get().split('|')
        ranking = round(1-int(ranks[0])/int(ranks[1]), 4)
        item = ManagerItem(fticker=fticker, ranking=ranking)
        print('='*30)
        yield item