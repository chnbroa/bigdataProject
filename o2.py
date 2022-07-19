import pandas as pd
import numpy as np
import os
import cv2
from pandas import Series, DataFrame
from bs4 import BeautifulSoup

site = "https://www.bigdatahub.go.kr/images/disease/"
data_name = ["검은별무늬병", "그을음병", "노균병"]
print(data_name[0])
data = pd.read_csv(f'data\{data_name[0]}.csv')
data.describe()
data.info()
file_name = data["파일명(File_Name)"]
Act_Diag = Series([], dtype="string")
Act_Diag = data["실제판독명(Act_Diag)"]
Act_Diag = Act_Diag.str.split('+')
file_list = list(zip(file_name, Act_Diag))
for i, j in file_list:
    if len(j) == 1:
        print(i, j)
        url = site+i
        os.system("curl " + url + f"> {i}")
