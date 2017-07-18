import urllib.request
from bs4 import BeautifulSoup

url = 'http://www.dajunzk.com/'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:53.0) Gecko/20100101 Firefox/53.0 ',
                      'host':'img01.zhaopin.cn'}
opener = urllib.request.build_opener()
opener.addheaders = [headers]
data = opener.open(url, timeout=30)
html = data.read()
soup = BeautifulSoup(html,'lxml')
for i in soup.find_all('a'):
    print(i.get('href'))