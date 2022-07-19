import requests
import os
import urllib.request


url = "https://www.bigdatahub.go.kr/images/disease/j1_145(7).jpg"

urllib.request.urlretrieve(url, 'test.jpg')
