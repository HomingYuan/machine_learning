#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Homing
@software: PyCharm Community Edition
@file: pandas+manual.py
@time: 2017/6/14 23:42
"""

import urllib.request
from bs4 import BeautifulSoup


def get_html(url):
    headers = {"Host": 'pandas.pydata.org',
               "User-Agebr": "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:53.0) Gecko/20100101 Firefox/53.0"}
    opener = urllib.request.build_opener()
    opener.addheaders = [headers]
    data = opener.open(url, timeout=30)
    html = data.read()
    data.close()
    return html

l = []
url = 'http://pandas.pydata.org/pandas-docs/stable'
soup = BeautifulSoup(get_html(url), 'lxml')
for item in soup.find_all('a'):
    if 'href' in item.attrs:
        l.append(item.get_text().strip())
        href = 'http://pandas.pydata.org/pandas-docs/stable/' + item.attrs['href']
        l.append(href)

fo = open('pandas_manual.csv', 'w', encoding='utf-8')


for i in range(len(l)):
    if i % 2 == 0:
        fo.write(l[i])
        fo.write(',')
    if i % 2 == 1:
        fo.write(l[i])
        fo.write('\n')






