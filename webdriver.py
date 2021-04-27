# from selenium import webdriver
from webbot import Browser
import time
# class northshoreSpider(Spider):
#     name = 'xxx'
#     allowed_domains = ['https://tuoitre.vn/']
#     start_urls = ['https://tuoitre.vn/']

#     def __init__(self):
#         self.driver = webdriver.Firefox()

#     def parse(self,response):
#             self.driver.get('https://tuoitre.vn/tim-kiem.htm?keywords=a')

#             while True:
#                 try:
#                     next = self.driver.find_element_by_xpath('//*[@class="btn-readmore"]')
#                     url = 'https://tuoitre.vn/tim-kiem.htm?keywords=a'
#                     yield Request(url,callback=self.parse2)
#                     next.click()
#                 except:
#                     break

#             self.driver.close()

#     def parse2(self,response):
#         print ('you are here!')

browser = Browser()
browser.go_to('https://tuoitre.vn/thuy-tien-giai-trinh-178-ti-cuu-tro-mien-trung-2-vo-chong-ca-si-gop-them-gan-3-7-ti-20201123215443513.htm')
time.sleep(10)
print('hi')
comment = browser.find_elements(classname="article-title")
print('hi')
print(comment)
comment.text
print(comment)