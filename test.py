# test

import pandas as pd
import numpy as np
import seaborn as sb
import sklearn as skl
import matplotlib as mpl # 그래프
import matplotlib.pyplot as plt # 그래프 관련
from tabulate import tabulate
from sklearn.preprocessing import MinMaxScaler,StandardScaler # 스케일러
from sklearn.preprocessing import LabelEncoder,OneHotEncoder # 인코더
from sklearn.preprocessing import LabelBinarizer # 멀티클래스 이진 변환
from sklearn.linear_model import LinearRegression,LogisticRegression # 모델 : 선형회귀, 로지스틱
from sklearn.tree import DecisionTreeClassifier # 모델 : 의사결정트리
from sklearn.ensemble import RandomForestClassifier # 모델 : 랜덤포레스트
from sklearn.svm import SVC # 모델 : SVC
from sklearn.datasets import make_classification # 무작위 데이터 생성기
from sklearn.model_selection import train_test_split # 훈련/평가 데이터 분리
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report # 평가 프로세스
from sklearn.metrics import roc_auc_score, roc_curve # ROC,AUC
from matplotlib import font_manager
from matplotlib import rc

mpl.rc('axes',unicode_minus=False)
font = font_manager.FontProperties(fname = "c:/Windows/Fonts/malgun.ttf").get_name()
rc('font',family=font)

def call(data,style=None):
  if style == None:
    print(tabulate(data,headers='keys',tablefmt='github'))
  else:
    print(tabulate(data,headers='keys',tablefmt=style))

def checker(data):
  try:
    if isinstance(data,pd.DataFrame):
      print('데이터 정보 출력')
      data.info()   
      print(f'행: {data.shape[0]}, 열: {data.shape[1]}')
      print('-'*50)
      print(data)
      return
    elif isinstance(data,list):
      data=pd.Series(data)
    print('데이터 정보 출력')
    data.info()
    print(f'행: {data.shape[0]}')
    print('-'*50)
    print(data)
    print(f'{'-'*50}\n{data.value_counts()}')
  except:
    print('>>> 경고! 데이터 형식이 잘못되었습니다!\n>>> checker(data) / repeat= 샘플 출력 횟수')

pathfind = 'C:/Mtest/project_first/pjdata/seoul/서울.csv'
path2 = 'C:/Mtest/project_first/pjdata/jeju-island/성산2.csv'

#--------------------------------------------------

# 컬럼명을 수동으로 지정해 1~7행 무시
columns = ["년월", "지점", "평균기온(℃)", "평균최저기온(℃)", "평균최고기온(℃)"]
data = pd.read_csv(pathfind, encoding="cp949", skiprows=8, names=columns)

# strip으로 공백제거
data["년월"] = data["년월"].str.strip()

# 25년 데이터 제외
testdata = data.tail(3) # 평가할 때 사용

data.drop(data.tail(3).index,inplace=True) # 방안 1

#data = data[~data["년월"].str.contains("2025", na=False)] # 방안 2

# data["년월"] = pd.to_datetime(data["년월"], format="%Y-%m") # 방안 3
# data["연도"] = data["년월"].dt.year # 연도만 추출
# data = data[data["연도"] != 2025]  # 2025년 데이터 제거
call(data.tail())

'''
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   년월         639 non-null    object 
 1   지점         639 non-null    int64  
 2   평균기온(℃)    638 non-null    float64
 3   평균최저기온(℃)  638 non-null    float64
 4   평균최고기온(℃)  638 non-null    float64

     년월       지점    평균기온(℃)    평균최저기온(℃)    평균최고기온(℃)
--  -------  ------  -------------  -----------------  -----------------
 0  1972-01     108            0.8               -2.1                4.5
 1  1972-02     108           -0.6               -3.5                3
 2  1972-03     108            5.3                1.4               10.3
 3  1972-04     108           11.5                6.9               16.7
 4  1972-05     108           16.4               11.7               21.3

행: 639, 열: 5
'''
for col in data.columns:
    print(f"data[{col}].isnull()", ':', data[col].isnull().sum())

print('='*80)
