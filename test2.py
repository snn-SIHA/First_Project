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

folder   = 'busan'
filename = '부산'
pathfind = f'C:/Mtest/project_first/data/{folder}/{filename}.csv'
pathsave = f'C:/Mtest/project_first/data/{folder}/{filename}refine.csv'
pathtest = f'C:/Mtest/project_first/data/{folder}/{filename}devide.csv'

#--------------------------------------------------

# 컬럼명을 수동으로 지정해 1~7행 무시
columns = ["년월", "지점", "평균기온(℃)", "평균최저기온(℃)", "평균최고기온(℃)"]
data = pd.read_csv(pathfind, encoding="cp949", skiprows=8, names=columns)

# strip으로 공백제거
data["년월"] = data["년월"].str.strip()

# 년월 컬럼 년도,월 분리 프로세스
data.년월 = pd.to_datetime(data.년월)
data.insert(1, '년도', data.년월.dt.year)
data.insert(2, '월', data.년월.dt.month)
data.drop('년월',axis=1,inplace=True)

# 2000년도부터 데이터프레임 재생성
data = data[data.년도>=2000]
data.reset_index(drop=True,inplace=True)

# 25년 데이터 분리
testdata = data[data.년도==2025] # 평가할 때 사용
testdata.reset_index(drop=True,inplace=True)

# 25년 데이터 제외
data.drop(data[data.년도 == 2025].index,inplace=True) # 방안 1

checker(data)
print('- '*40)
call(data.head(),'plane')
call(data.tail(),'plane')
print('- '*40)

call(testdata,'plane')
#data = data[~data["년월"].str.contains("2025", na=False)] # 방안 2

# data["년월"] = pd.to_datetime(data["년월"], format="%Y-%m") # 방안 3
# data["연도"] = data["년월"].dt.year # 연도만 추출
# data = data[data["연도"] != 2025]  # 2025년 데이터 제거
#print(data.head())

data.to_csv(pathsave,index=False,encoding='cp949')
testdata.to_csv(pathtest,index=False,encoding='cp949')

print('='*80)
