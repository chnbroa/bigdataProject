import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import os
import urllib.request
import requests

site = "https://www.bigdatahub.go.kr/images/disease/"
data_name = ["검은별무늬병", "그을음병", "노균병","덩굴마름병","모자이크병","세균성모무늬병","탄저병","흰가루병","정상"]
data_test=data_name[7]
print(f'{data_test}')
data = pd.read_csv('data\{}.csv'.format(data_test))
data.describe()
data.info()

file_name = data["파일명(File_Name)"]
Act_Diag = Series([],dtype="string")
Act_Diag = data["실제판독명(Act_Diag)"]
Act_Diag= Act_Diag.str.split('+')
file_list= list(zip(file_name,Act_Diag))

PATH ="C:/data/{}".format(data_test) #저장 경로
print(PATH)
if not os.path.isdir(PATH): #폴더가 존재하지 않는다면 폴더 생성
    os.makedirs(PATH)

for i, j in file_list:
    try:
        if len(j)== 1:
            print(i, j)
            url = site+i
            urllib.request.urlretrieve(url,'{}/{}'.format(PATH,i))
            print('download successful')
    except:
        print("오류발생!!")
print('finish!')
#https://www.delftstack.com/ko/howto/python/download-image-in-python/
