# predict_model
# 각종 파일과 모델을 한 파일에서 실행할 수 있도록 구성된 코드.
# 코드 실행 후 폴더명, 파일명을 입력하여 csv파일 열람.
# 이후 모델을 입력하여 학습 후 예측 실행
# 본 문서에서는 2000~2023년 까지의 데이터를 학습하고 2024년을 예측.

'''
모델 구성
LNR : 선형회귀(LinearRegression)
DTR : 의사결정트리(DecisionTreeRegressor)
RFR : 랜덤포레스트(RandomForestRegressor)
SVM : 서포트벡터(SVR)
'''

#--------------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sb
import sklearn as skl
import matplotlib as mpl # 그래프
import matplotlib.pyplot as plt # 그래프 관련
from matplotlib.gridspec import GridSpec # 그래프 관련
from sklearn.preprocessing import LabelEncoder # 인코더
from sklearn.preprocessing import StandardScaler # 스케일러
from sklearn.linear_model import LinearRegression # 선형회귀
from sklearn.tree import DecisionTreeRegressor # 의사결정트리
from sklearn.ensemble import RandomForestRegressor # 랜덤포레스트
from sklearn.svm import SVR # 서포트벡터
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

pathsave = 'C:/Mtest/project_first/'

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
#데이터 분리 : 2000~2023년 : 훈련 / 2024년 : 평가

dftr = df1[df1.년도<=2023]
dfte = df1[df1.년도==2024]

#--------------------------------------------------
# 모델 선정
LNR = LinearRegression()
DTR = DecisionTreeRegressor()
RFR = RandomForestRegressor()
SVM = SVR(kernel='linear')

# 인코딩
LBE = LabelEncoder()
dftr.loc[:, '계절'] = LBE.fit_transform(dftr.계절)
dfte.loc[:, '계절'] = LBE.transform(dfte.계절)

# 고려사항 : 스케일링 여부? 그냥 필수로 때려박겠슴다.
STS = StandardScaler()

Xtrain = dftr[['년도','월','지점','계절','sin_month','cos_month']]
Xtest = dfte[['년도','월','지점','계절','sin_month','cos_month']]
ytrain = dftr[['평균기온(℃)']]
ytest = dfte[['평균기온(℃)']]

ytrain = STS.fit_transform(ytrain)
ytest = STS.transform(ytest)

print(f'훈련 데이터 규모 : {Xtrain.shape[0]}')
print(f'평가 데이터 규모 : {Xtest.shape[0]}')
print('- '*40)

#--------------------------------------------------
try:
  model = eval(input('>>> 모델 입력 : '))
except:
  print('>>> 모델명 입력 오류')

#--------------------------------------------------
# 모델 학습/예측 진행
model.fit(Xtrain,ytrain.ravel()) # 경고 출력
pre = model.predict(Xtest).reshape(-1,1)

# 평가 진행
mae = mean_absolute_error(ytest,pre)
mse = mean_squared_error(ytest,pre)
rmse = root_mean_squared_error(ytest,pre)
r2 = r2_score(ytest,pre)

# 스케일링 변환 후 터미널 결과 출력
print(f'실제\n{STS.inverse_transform(ytest)}\n')
print(f'예측\n{STS.inverse_transform(pre).round(1)}')
print('- '*40)

#--------------------------------------------------
# 결과 데이터프레임화
df1 = pd.DataFrame(STS.inverse_transform(ytest))        # 실제값
df2 = pd.DataFrame(STS.inverse_transform(pre).round(1)) # 예측값
df = dfte[['년도','월','계절']].reset_index(drop=True)         # 년,월
df['실제 평균온도'] = df1
df['예측 평균온도'] = df2
df['예측 편차'] = df2-df1
df.계절 = LBE.inverse_transform(df.계절.astype('int')) # 계절 인코딩 풀기
print(df)
print('- '*40)

#--------------------------------------------------
# 결과 종합
print(f'\n사용된 모델 : {model}')
print(f'MAE : {mae}\nMSE : {mse}\nRMSE : {rmse}\nR2_score : {r2}')

# 시각화 1 : 실제 온도 / 예측 온도 비교 그래프
season = df.계절.unique()

fig,ax = plt.subplots(2,2,figsize=(12,8))
for i,s in enumerate(season):
  dfs = df[df.계절==s]
  if s == '겨울':
    dfs.loc[dfs['월']==12,'월'] = 0
  row,col = i//2,i%2 # 각 그래프의 행과 열
  sb.lineplot(dfs,x='월',y='실제 평균온도',label='실제 평균온도',ax=ax[row,col],color='royalblue')
  sb.lineplot(dfs,x='월',y='예측 평균온도',label='예측 평균온도',ax=ax[row,col],color='red')
  ax[row,col].set_title(f'2024년 {s} 기온 예측 결과')
  ax[row,col].legend()
plt.tight_layout()
plt.savefig(pathsave+'test1')
# plt.show()

# 시각화 2 : 계절별 온도 그래프? 아니면... 흠...
# 여러 모델을 동시에 돌리는 코드도 만들 거라면, 차라리 R2를 비교하는 그래프?
# 일단 오늘은 여기까지 :p...
fig = plt.figure(figsize=(12,8))
gs = GridSpec(2,2,figure=fig)

ax1=fig.add_subplot(gs[0,0])

# plt.show()

print('='*80)
