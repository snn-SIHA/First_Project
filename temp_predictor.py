# temp_predictor
# ver 1.3.1
# 기본 구성은 predict_2025와 동일
# 본 문서에서는 while문과 터미널을 활용한 인터페이스 구성을 시험
# 그래픽 출력 수정
# 출력 속도 제한 적용

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
from sklearn.svm import SVR # 서포트 벡터 회귀
from matplotlib import font_manager
from matplotlib import rc
import time
import os
import warnings

mpl.rc('axes', unicode_minus=False)
mpl.rcParams['axes.unicode_minus']=False
pd.set_option("display.max_colwidth",20) # 출력할 열의 너비
pd.set_option("display.unicode.east_asian_width",True) # 유니코드 사용 너비 조정
font_name = font_manager.FontProperties(fname='c:/windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)
warnings.simplefilter("ignore", UserWarning)

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
    print('\n>>> 경고! 데이터를 호출할 수 없습니다!\n>>> 폴더 경로를 확인하거나, 파일이 잘못되었을 수 있습니다.')
    return None,None,None,None

#--------------------------------------------------
# 프로그램 소개
intro = f'''
{'- '*40}
이 프로그램은 2000년부터 2024년까지의 기상청 데이터를 바탕으로 2025년의 온도를 예측합니다.
기본적으로 ./data 폴더를 기반으로 데이터를 불러오며, 폴더와 파일명을 입력하는 것으로 데이터 파일을 호출할 수 있습니다.
기본 설정으로 랜덤포레스트 회귀 모델을 사용합니다. 모델은 사용자가 변경할 수 있으며, 변경 가능한 모델은 선형회귀, 의사결정트리, 랜덤포레스트, 서포트벡터회귀 모델을 지원합니다.
필요에 따라 예측 후 그래프를 생성하고 저장할 수 있으며, 그래프는 ./pic 폴더에 저장됩니다. 또한, 저장 후 즉시 출력하는 기능이 포함되어 있습니다.
{'- '*40}
'''

#--------------------------------------------------
# 예측 실행 함수
def model_running(m):
  model = m
  # 데이터프레임 호출
  df1,df2,locate_folder,locate_file = load_dataframe() # 폴더명, 파일명으로 추적 
  if df1 is None:
    print('- '*40)
    return  # 입력 오류시 예측 중단
  print('- '*40)
  checker(df1) # 데이터프레임 출력
  time.sleep(0.8)
  print(f'{'- '*40}\n>>> 데이터 확인 성공, 학습 준비 중.')
  time.sleep(0.5)

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

  # 계절성 주기 특성 추가
  df1['sin_month'] = np.sin(2 * np.pi * df1['월'] / 12)  # 사인 함수로 주기성 표현
  df1['cos_month'] = np.cos(2 * np.pi * df1['월'] / 12)  # 코사인 함수로 주기성 표현
  df2['sin_month'] = np.sin(2 * np.pi * df2['월'] / 12)
  df2['cos_month'] = np.cos(2 * np.pi * df2['월'] / 12)

  #--------------------------------------------------
  # 인코딩
  LBE = LabelEncoder()
  df1.loc[:, '계절'] = LBE.fit_transform(df1.계절)
  df2.loc[:, '계절'] = LBE.transform(df2.계절)

  # 훈련, 평가 데이터 분리
  Xtrain = df1[['년도','월','지점','계절','sin_month','cos_month']]
  Xtest = df2[['년도','월','지점','계절','sin_month','cos_month']]
  ytrain = df1[['평균기온(℃)']]
  ytest = df2[['평균기온(℃)']]

  # 스케일링
  STS = StandardScaler()
  ytrain = STS.fit_transform(ytrain)
  ytest = STS.transform(ytest)
  
  print(f'>>> 학습 진행 중.')
  time.sleep(0.4)

  #--------------------------------------------------
  # 모델 학습/예측 진행
  model.fit(Xtrain,ytrain.ravel())
  pre = model.predict(Xtest).reshape(-1,1)
  
  print(f'>>> 예측 진행 중.\n{'- '*40}')
  time.sleep(0.25)

  #--------------------------------------------------
  # 결과 데이터프레임화
  dfp = pd.DataFrame(STS.inverse_transform(pre).round(1)) # 예측값
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
  print(f'사용된 모델 : {model}\n훈련 데이터 규모 : {Xtrain.shape[0]}\n평가 데이터 규모 : {Xtest.shape[0]}\n예측 지역 : {locate_file}\n{'- '*40}')
  print(dfp)

  #--------------------------------------------------
  # 시각화 그래프 생성 여부
  while True:
    gcode = input('\n>>> 그래프를 생성하시겠습니까? (y/n) : ')
    if gcode == 'y':
      os.makedirs("./pic", exist_ok=True) # 폴더 유무 체크 후 생성
      # 시각화 : lineplot
      fig1 = plt.figure(figsize=(8,4.5))
      plt.plot(cal['월'],cal['평균기온(℃)'],marker='o',markersize=4.5,label='2024년 평균온도',color='royalblue')
      plt.plot(dfp['월'],dfp['평균기온(℃)'],marker='o',markersize=4.5,label='2025년 평균온도',color='forestgreen')
      plt.plot(cal['월'],cal['예측기온(℃)'],marker='o',markersize=4.5,label='2025년 예측온도',color='red')

      for i, (x, y1, y2) in enumerate(zip(cal['월'], cal['평균기온(℃)'], cal['예측기온(℃)'])):
        plt.text(x, y1 + 1.5, f"{y1:.1f}", ha='center', fontsize=9, color='royalblue')
        plt.text(x, y2 - 2.2, f"{y2:.1f}", ha='center', fontsize=9, color='red')

      plt.xticks(range(1, 13), labels=[f"{i}월" for i in range(1, 13)])
      plt.ylabel('온도(℃)')
      plt.xlabel('')
      plt.ylim(-3.5,32.5)
      plt.title(f'2025년 {locate_file} 기온 예측\n')
      plt.text(6.5, 33.5, f'예측 모델 : {model}', fontsize=9.5, ha='center')
      plt.legend(loc='best')
      plt.grid(axis='y',linestyle='--',alpha=0.35)
      plt.tight_layout()
      plt.savefig(f'C:/Mtest/project_first/pic/{locate_folder}_2025_L_1.0.png')
      print(f'>>> C:/Mtest/project_first/pic/{locate_folder}_2025_L_1.0.png')
      #--------------------------------------------------
      # 시각화 : barplot
      fig2 = plt.figure(figsize=(8,4.5))
      colors = ['deepskyblue' if y > 0 else 'orange' for y in dfp['전년대비']]
      sb.barplot(dfp,x='월',y='전년대비',palette=colors,hue='월',legend=False)
      
      for i, (x, y) in enumerate(zip(dfp['월'], dfp['전년대비'])):  # '기온변화' 컬럼 사용
        offset = 0.05 if y > 0 else -0.12
        plt.text(i, y+offset, f"{y:.1f}", fontsize=9, ha='center', color='deepskyblue' if y > 0 else 'orange')
      
      plt.xticks(range(12), labels=[f"{i}월" for i in range(1, 13)])
      plt.ylabel('전년 대비 온도 편차')
      plt.xlabel('')

      y_abs_max = max(abs(dfp['전년대비'].min()), abs(dfp['전년대비'].max()))
      ylim_max = max(1.5, y_abs_max)
      ylim_max = min(ylim_max+(y_abs_max*0.15), 4.5)
      ylim_min = -ylim_max

      plt.ylim(ylim_min, ylim_max)
      plt.title(f'2025년 {locate_file} 전년 대비 온도 변화\n')
      sub_y = ylim_max * 1.05
      plt.text(5.5, sub_y, f'예측 모델 : {model}', fontsize=9.5, ha='center')
      plt.grid(axis='y',linestyle='--',alpha=0.35)
      plt.tight_layout()
      plt.savefig(f'C:/Mtest/project_first/pic/{locate_folder}_2025_B_1.0.png')
      print(f'>>> C:/Mtest/project_first/pic/{locate_folder}_2025_B_1.0.png')
      print(f'>>> 그래프 생성이 완료되었습니다.\n')
      while True:
        gcode = input('>>> 그래프를 바로 확인하시겠습니까? (y/n) : ')
        if gcode == 'y':
          print('- '*40)
          def on_close(event):
            plt.close('all')
          fig1.canvas.mpl_connect('close_event', on_close)
          fig2.canvas.mpl_connect('close_event', on_close)
          plt.show()
          break
        elif gcode == 'n':
          print('- '*40)
          break
        else: print('>>> 다시 입력해주세요.\n')
      break
    elif gcode == 'n':
      print(f'>>> 그래프 생성을 취소합니다.\n{'- '*40}')
      break
    else:
      print('>>> 경고! 잘못된 입력입니다.\n')

#--------------------------------------------------
# 프로그램
def predictor():
  LNR = LinearRegression()
  DTR = DecisionTreeRegressor()
  RFR = RandomForestRegressor()
  SVM = SVR(kernel='linear')
  model = RFR # 기본 모델 : 랜덤포레스트회귀
  print('>>> 2025년 기온 예측 프로그램을 실행합니다.')
  while True:
    print('>>> 1. 예측 실행 / 2. 예측 모델 확인 / 3. 모델 변경 / 4. 프로그램 소개 / q. 종료')
    exec = input('>>> 실행 작업 선택 : ')

    if exec == '1':
      print(f'\n>>> 예측 모델을 실행합니다.')
      model_running(model)

    elif exec == '2':print(f'\n>>> 현재 적용된 모델을 확인합니다.\n현재 모델 : {model}\n{'- '*40}')

    elif exec == '3':
      print('\n>>> 예측 모델을 변경합니다.')
      try:
        print(">>> 사용 가능한 모델 목록\nLNR : LinearRegression()\nDTR : DecisionTreeRegressor()\nRFR : RandomForestRegressor()\nSVM : SVR(kernel='linear')\n")
        model = eval(input('>>> 모델 입력 : '))
        print(f'>>> 모델 변경 완료.\n{'- '*40}')
      except:
        print(f'>>> 모델명 입력 오류\n{'- '*40}')

    elif exec == '4':print(intro)

    elif exec == 'q':
      print(f'\n>>> 프로그램을 종료합니다.')
      break

    else:
      print('>>> 경고! 잘못된 입력입니다.\n')
      continue

if __name__ == "__main__":
  predictor()

print('='*80)
