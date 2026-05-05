from bs4 import BeautifulSoup
import requests

try:
    response = requests.get('https://modelscope.cn/models?search=openvino')
    soup = BeautifulSoup(response.text, 'html.parser')
    for a in soup.find_all('a'):
        href = a.get('href')
        if href and href.startswith('/models/'):
            print(href)
except Exception as e:
    print(e)
