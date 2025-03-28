# test_LNR
# 선형회귀 모델 사용

import pandas as pd
import numpy as np
import seaborn as sb
import sklearn as skl
import matplotlib as mpl # 그래프
import matplotlib.pyplot as plt # 그래프 관련
from sklearn.preprocessing import LabelEncoder # 인코더
from sklearn.preprocessing import MinMaxScaler,StandardScaler # 스케일러
from sklearn.linear_model import LinearRegression,LogisticRegression # 모델 : 선형회귀, 로지스틱
from sklearn.tree import DecisionTreeRegressor # 모델 : 의사결정트리
from sklearn.ensemble import RandomForestRegressor # 모델 : 랜덤포레스트
from sklearn.svm import SVR # 모델 : 서포트벡터
from sklearn.model_selection import train_test_split # 훈련/평가 데이터 분리
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score # 평가 프로세스
from sklearn.metrics import roc_auc_score, roc_curve # ROC,AUC
from matplotlib import font_manager
from matplotlib import rc

mpl.rc('axes', unicode_minus=False)
mpl.rcParams['axes.unicode_minus']=False
pd.set_option("display.max_colwidth",20) # 출력할 열의 너비
pd.set_option("display.unicode.east_asian_width",True) # 유니코드 사용 너비 조정
font_name = font_manager.FontProperties(fname='c:/windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)

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

folder   = 'seoul'
filename = '서울'
pathre = f'C:/Mtest/project_first/data/{folder}/{filename}refine.csv'
pathde = f'C:/Mtest/project_first/data/{folder}/{filename}devide.csv'

#--------------------------------------------------
df1 = pd.read_csv(pathre, encoding='cp949')
df2 = pd.read_csv(pathde, encoding='cp949')

# 전월대비온도변화 특성 추가
df1['전월대비'] = df1['평균기온(℃)'].diff()
df1['전월대비'] = df1.전월대비.fillna(0)
df2['전월대비'] = df2['평균기온(℃)'].diff()
df2['전월대비'] = df2.전월대비.fillna(0)

# 작년대비온도변화 특성 추가
df1['작년대비'] = df1['평균기온(℃)'].diff(12)
df1['작년대비'] = df1.작년대비.fillna(0)
df2['작년대비'] = df2['평균기온(℃)'].diff(12)
df2['작년대비'] = df2.작년대비.fillna(0)

# 계절 구분 특성 추가
season_list = {
  '봄':[3,4,5],
  '여름':[6,7,8],
  '가을':[9,10,11],
  '겨울':[12,1,2]
}

def season_executor(month):
  for rst,search in season_list.items():
    if month in search:
      return rst
  return 'ERROR'

df1['계절'] = df1.월.apply(season_executor)
df2['계절'] = df2.월.apply(season_executor)

# GPT : 계절성주기 되시겠습니다
df1['sin_month'] = np.sin(2 * np.pi * df1['월'] / 12)  # 사인 함수로 주기성 표현
df1['cos_month'] = np.cos(2 * np.pi * df1['월'] / 12)  # 코사인 함수로 주기성 표현
df2['sin_month'] = np.sin(2 * np.pi * df2['월'] / 12)
df2['cos_month'] = np.cos(2 * np.pi * df2['월'] / 12)

checker(df1)
print('- '*40)

'''
컬럼명:
년도/월/지점/평균기온(℃)/평균최저기온(℃)/평균최고기온(℃)/전월대비/작년대비/계절/sin_month/cos_month
'''
#--------------------------------------------------

LNR = LinearRegression()
SVM = SVR() #?
DTR = DecisionTreeRegressor() # 이거 괜찮을 지도.
RFR = RandomForestRegressor()

# 인코딩
LBE = LabelEncoder()
df1.계절 = LBE.fit_transform(df1.계절)
df2.계절 = LBE.transform(df2.계절)

# 고려사항 : 스케일링 여부? 흠.
STS = StandardScaler()

Xtrain = df1[['년도','월','지점','계절','sin_month','cos_month']]
Xtest = df2[['년도','월','지점','계절','sin_month','cos_month']].drop(df2.index[-1])
ytrain = df1[['평균기온(℃)','평균최저기온(℃)','평균최고기온(℃)','전월대비','작년대비']]
ytest = df2[['평균기온(℃)','평균최저기온(℃)','평균최고기온(℃)','전월대비','작년대비']].drop(df2.index[-1])

ytrain = STS.fit_transform(ytrain)
ytest = STS.transform(ytest)

print(f'훈련 데이터 규모 : {Xtrain.shape[0]}')
print('- '*40)

#--------------------------------------------------
model = LNR # 모델 입력

#--------------------------------------------------
model.fit(Xtrain,ytrain)
pre = model.predict(Xtest)

mae = mean_absolute_error(ytest,pre)
mse = mean_squared_error(ytest,pre)
rmse = root_mean_squared_error(ytest,pre)
r2 = r2_score(ytest,pre)

print(f'실제\n{ytest}\n')
print(f'예측\n{pre}')
print('- '*40)
print(f'실제\n{STS.inverse_transform(ytest)}\n')
print(f'예측\n{STS.inverse_transform(pre).round(1)}')

print('='*80)
