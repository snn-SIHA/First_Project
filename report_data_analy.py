# report_data_analysis
# 모델별 평가결과 분석

import pandas as pd
import numpy as np
import seaborn as sb
import sklearn as skl
import matplotlib as mpl # 그래프
import matplotlib.pyplot as plt # 그래프 관련
from matplotlib.gridspec import GridSpec # 그래프 관련
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

#--------------------------------------------------
# 평가 보고서 출력
df = pd.read_csv('C:/Mtest/project_first/report_data.csv',encoding='cp949')
checker(df)
print('- '*40)

print(df[df.모델=='DTR'])
print('- '*40)

print(df[df.지역=='서울'])
print('- '*40)

print(f'DTR 평균 : {df[df.모델=='DTR'].R2_score.mean()}')
print(f'LNR 평균 : {df[df.모델=='LNR'].R2_score.mean()}')
print(f'RFR 평균 : {df[df.모델=='RFR'].R2_score.mean()}')
print(f'SVR 평균 : {df[df.모델=='SVR'].R2_score.mean()}')

#--------------------------------------------------
# 시각화 준비
r2_avg = df.groupby("모델")["R2_score"].mean().round(3) # 평균 계산
model = df.모델.unique()
colors = sb.color_palette("pastel", len(model)) # 파스텔 색상 추출

# 시각화 : LNR,SVR
fig = plt.figure(figsize=(10,5)) # figure
gs = GridSpec(2,5,figure=fig) # gridspec (그래프 세부 설정을 위해)

for i,m in enumerate(['LNR','SVR']):
  ax = fig.add_subplot(gs[i,3:5])
  dfm = df[df.모델==m]
  sb.lineplot(dfm,x='지역',y='R2_score',ax=ax,color=colors[i])
  ax.set_ylabel('R2 Score')
  ax.set_title(f'{m} 예측 R2_score')
  ax.set_ylim(0.9, 1.0)
  ax.set_xlabel('')
  ax.set_ylabel('')

main=fig.add_subplot(gs[:,:3])
sb.barplot(x=r2_avg[['LNR','SVR']].index,y=r2_avg[['LNR','SVR']].values,ax=main,hue=r2_avg[['LNR', 'SVR']].index,palette='pastel')
main.set_ylim(0.85, 1.0)
main.set_title("모델 별 예측 R2_score 평균")
main.set_ylabel("R2 Score")
for i, v in enumerate(r2_avg[['LNR','SVR']].values):
  main.text(i, v+0.002, f'{v:.4f}', ha='center', va='bottom', fontsize=10.5)
plt.tight_layout()
plt.savefig('C:/Mtest/project_first/report_1.png')
plt.show()

#--------------------------------------------------
# 시각화 : DTR,RFR
fig = plt.figure(figsize=(10,5))
gs = GridSpec(2,5,figure=fig)

for i,m in enumerate(['DTR','RFR']):
  ax = fig.add_subplot(gs[i,3:5])
  dfm = df[df.모델==m]
  sb.lineplot(dfm,x='지역',y='R2_score',ax=ax,color=colors[i])
  ax.set_ylabel('R2 Score')
  ax.set_title(f'{m} 예측 R2_score')
  ax.set_ylim(0.9, 1.0)
  ax.set_xlabel('')
  ax.set_ylabel('')

main=fig.add_subplot(gs[:,:3])
sb.barplot(x=r2_avg[['DTR','RFR']].index,y=r2_avg[['DTR','RFR']].values,ax=main,hue=r2_avg[['DTR','RFR']].index,palette='pastel')
main.set_ylim(0.85, 1.0)
main.set_title("모델 별 예측 R2_score 평균")
main.set_ylabel("R2 Score")
for i, v in enumerate(r2_avg[['DTR','RFR']].values):
  main.text(i, v+0.002, f'{v:.4f}', ha='center', va='bottom', fontsize=10.5)
plt.tight_layout()
plt.savefig('C:/Mtest/project_first/report_2.png')
plt.show()

print('='*80)
