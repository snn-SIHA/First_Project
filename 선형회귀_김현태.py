# test_LNR
# 선형회귀 모델 사용

import pandas as pd
import numpy as np
import seaborn as sb
import sklearn as skl
import matplotlib as mpl # 그래프
import matplotlib.pyplot as plt # 그래프 관련
from sklearn.preprocessing import LabelEncoder # 인코더
from sklearn.preprocessing import StandardScaler # 스케일러
from sklearn.linear_model import LinearRegression,LogisticRegression # 모델 : 선형회귀, 로지스틱
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score # 평가 프로세스
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

def load_dataframe(name_folder=None,name_file=None,encodeing_option='cp949'):
  try:
    if name_folder is None:
      name_folder = input('>>> 데이터를 불러올 폴더명을 입력하세요 : ')
      name_file = input('>>> 데이터를 불러올 파일명을 입력하세요 : ')
    pathfind = f'C:/Mtest/project_first/data/{name_folder}/{name_file}'
    dataframe_refine = pd.read_csv(pathfind+'refine.csv', encoding=encodeing_option)
    dataframe_devide = pd.read_csv(pathfind+'devide.csv', encoding=encodeing_option)
    return dataframe_refine,dataframe_devide
  except:
    print('>>> 경고! 데이터를 호출할 수 없습니다!\n>>> 폴더 경로를 확인하거나, 파일이 잘못되었을 수 있습니다.')
    return None,None

#--------------------------------------------------
df1,df2 = load_dataframe('seoul','서울')

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

# 모델 선정
LNR = LinearRegression()

# 인코딩
LBE = LabelEncoder()
df1.계절 = LBE.fit_transform(df1.계절)
df2.계절 = LBE.transform(df2.계절)

# 고려사항 : 스케일링 여부?
# 선형회귀는 스케일링이 크게 갈리므로 사용
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

print(f'\nMAE : {mae}\nMSE : {mse}\nRMSE : {rmse}\nR2_score : {r2}')
# R2_score 문제 많음 <-- 예측한 결과값이 너무 적은 탓

print('='*80)
