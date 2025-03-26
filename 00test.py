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

pathfind = 'C:/Mtest/project_first/pjdata/소방청_전국 산악사고 구조활동현황_20201231.csv'

#--------------------------------------------------

df = pd.read_csv(pathfind,encoding='euc-kr')
checker(df)
'''
 #   Column        Non-Null Count  Dtype 
---  ------        --------------  ----- 
 0   신고년월일         13189 non-null  object
 1   신고시각          13189 non-null  object
 2   출동년월일         13029 non-null  object
 3   출동시각          13029 non-null  object
 4   발생장소_시        13189 non-null  object
 5   발생장소_구        13189 non-null  object
 6   발생장소_동        13189 non-null  object
 7   발생장소_리        13189 non-null  object
 8   번지            13090 non-null  object
 9   사고원인          13189 non-null  object
 10  사고원인코드명_사고종별  13189 non-null  object
 11  구조인원          13189 non-null  int64
 행: 13189, 열: 12
'''

print('='*80)
