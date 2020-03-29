# -*- coding: utf-8 -*-


from scrapy import signals
from selenium import webdriver
import time
from scrapy.http import HtmlResponse


class EastmDownloaderMiddleware(object):

#     def __init__(self):
    driver_path = r'G:\Python\chromedriver.exe'
    driver = webdriver.Chrome(executable_path=driver_path)

        
    def process_request(self, request, spider):
        len_ = len('<GET http://fundf10.eastmoney.com/')
        if str(request)[len_:len_+4] == 'hytz':
            self.driver.get(request.url)
            time.sleep(1)
            source = self.driver.page_source
            response = HtmlResponse(url=self.driver.current_url, body=source, request=request,encoding='utf-8')
            return response
        else:
            return None