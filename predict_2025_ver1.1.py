# predict_2025
# ver 1.1
# 업데이트 내용 : 학습 데이터에 전년대비 추가
# 본 문서에서는 2000~2024년 까지의 데이터를 학습하고 2025년을 예측.

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
from sklearn.ensemble import RandomForestRegressor # 랜덤포레스트
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
    return dataframe_refine,dataframe_devide,name_folder,name_file
  except:
    print('>>> 경고! 데이터를 호출할 수 없습니다!\n>>> 폴더 경로를 확인하거나, 파일이 잘못되었을 수 있습니다.')
    return None,None,None,None

pathsave = 'C:/Mtest/project_first/'

#--------------------------------------------------
# 데이터프레임 호출
df1,df2,locate_folder,locate_file = load_dataframe() # 폴더명, 파일명으로 추적
print('- '*40)
checker(df1) # 데이터프레임 출력
print('- '*40)

#--------------------------------------------------
# 2025년 데이터 확장 준비
months = list(range(4, 13)) # 4월부터 12월까지
stations = df2["지점"].unique() # 25년 데이터에서 지역 식별번호 추출

# 2025년 데이터프레임 확장내용 생성
new_data = []
for station in stations:
  for month in months:
    new_data.append([2025, month, station, np.nan, np.nan, np.nan])
new_df = pd.DataFrame(new_data, columns=["년도","월","지점","평균기온(℃)","평균최저기온(℃)","평균최고기온(℃)"])

df2 = pd.concat([df2,new_df],ignore_index=True) # 25년 데이터를 1월부터 12월까지 병합

#--------------------------------------------------
# 전월대비온도변화 특성 추가
df1['전월대비'] = df1['평균기온(℃)'].diff()
df1['전월대비'] = df1.전월대비.fillna(0)
df2['전월대비'] = df2['평균기온(℃)'].diff()
df2['전월대비'] = df2.전월대비.fillna(0)

# 작년대비온도변화 특성 추가
df1['전년대비'] = df1['평균기온(℃)'].diff(12)
df1['전년대비'] = df1.전년대비.fillna(0)
df2['전년대비'] = df2['평균기온(℃)'].diff(12)
df2['전년대비'] = df2.전년대비.fillna(0)

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

# 계절성 주기 특성 추가
df1['sin_month'] = np.sin(2 * np.pi * df1['월'] / 12)  # 사인 함수로 주기성 표현
df1['cos_month'] = np.cos(2 * np.pi * df1['월'] / 12)  # 코사인 함수로 주기성 표현
df2['sin_month'] = np.sin(2 * np.pi * df2['월'] / 12)
df2['cos_month'] = np.cos(2 * np.pi * df2['월'] / 12)

print(f'>>> TRACER\n{df1}')
#--------------------------------------------------
# 모델 등록
RFR = RandomForestRegressor()

# 인코딩
LBE = LabelEncoder()
df1.loc[:, '계절'] = LBE.fit_transform(df1.계절)
df2.loc[:, '계절'] = LBE.transform(df2.계절)

# 훈련, 평가 데이터 분리
Xtrain = df1[['년도','월','지점','계절','sin_month','cos_month']]
Xtest = df2[['년도','월','지점','계절','sin_month','cos_month']]
ytrain = df1[['평균기온(℃)','전년대비']]
ytest = df2[['평균기온(℃)','전년대비']]

# 스케일링
STS = StandardScaler()
ytrain = STS.fit_transform(ytrain)
ytest = STS.transform(ytest)

# 훈련, 평가 데이터 규모 출력
print(f'예측 모델 데이터 관련 정보\n훈련 데이터 규모 : {Xtrain.shape[0]}\n평가 데이터 규모 : {Xtest.shape[0]}')
print('- '*40)

#--------------------------------------------------
# 모델 학습/예측 진행
RFR.fit(Xtrain,ytrain)
pre = RFR.predict(Xtest)

'''
# 평가 금지 : 2025년은 평가할 수 없음
mae = mean_absolute_error(ytest,pre)
mse = mean_squared_error(ytest,pre)
rmse = root_mean_squared_error(ytest,pre)
r2 = r2_score(ytest,pre)
'''
#--------------------------------------------------
# 결과 데이터프레임화
dfp = pd.DataFrame(STS.inverse_transform(pre).round(1)) # 예측값
dfp = dfp.iloc[:,0]
print(f'>>> TRACER\n{dfp}')
df2['예측기온(℃)'] = dfp

# 예측 결과 반영된 데이터프레임 생성
dfp = df2[['년도','월','계절','평균기온(℃)','예측기온(℃)']]
dfp = dfp.copy()
dfp.계절 = LBE.inverse_transform(dfp.계절.astype(int))

# 2025 전년대비를 구하기 위한 데이터프레임 
cal = df1[df1.년도==2024].copy().reset_index(drop=True)
cal['예측기온(℃)'] = dfp['예측기온(℃)']
cal['2025'] = cal['예측기온(℃)'] - cal['평균기온(℃)']
dfp['전년대비'] = cal['2025']

#--------------------------------------------------
# 결과 종합
print(f'사용된 모델 : {RFR}')
print(dfp)

#--------------------------------------------------
# 시각화 : lineplot
plt.figure(figsize=(8,4.5))
plt.plot(cal['월'],cal['평균기온(℃)'],marker='o',markersize=4.5,label='2024년 평균온도',color='royalblue')
plt.plot(dfp['월'],dfp['평균기온(℃)'],marker='o',markersize=4.5,label='2025년 평균온도',color='forestgreen')
plt.plot(cal['월'],cal['예측기온(℃)'],marker='o',markersize=4.5,label='2025년 예측온도',color='red')

# 세부 수치 추가
for i, (x, y1, y2) in enumerate(zip(cal['월'], cal['평균기온(℃)'], cal['예측기온(℃)'])):
  plt.text(x, y1 + 1.5, f"{y1:.1f}", ha='center', fontsize=9, color='royalblue')
  plt.text(x, y2 - 2.2, f"{y2:.1f}", ha='center', fontsize=9, color='red')

# 세부 설정
plt.xticks(range(1, 13), labels=[f"{i}월" for i in range(1, 13)])
plt.ylabel('온도(℃)')
plt.xlabel('')
plt.ylim(-3.5,32.5)
plt.title(f'2025년 {locate_file} 기온 예측')
plt.legend(loc='best')
plt.grid(axis='y',linestyle='--',alpha=0.35)
plt.tight_layout()
plt.savefig(f'C:/Mtest/project_first/{locate_folder}_2025_L_test.png')
plt.show()

#--------------------------------------------------
# 시각화 : barplot
plt.figure(figsize=(8,4.5))
sb.barplot(dfp,x='월',y='전년대비',color='orange',alpha=0.75)
plt.xticks(range(12), labels=[f"{i}월" for i in range(1, 13)])
plt.ylabel('온도(℃)')
plt.xlabel('')
plt.ylim(-1.5,1.5)
plt.title(f'2025년 {locate_file} 전년 대비 온도 변화')
plt.grid(axis='y',linestyle='--',alpha=0.35)
plt.tight_layout()
plt.savefig(f'C:/Mtest/project_first/{locate_folder}_2025_B_test.png')
plt.show()

print('='*80)
