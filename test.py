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

path2020 = 'C:/Mtest/project_first/pjdata/소방청_전국 산악사고 구조활동현황_20201231.csv'
path2023 = 'C:/Mtest/project_first/pjdata/소방청_전국 산악사고 현황_20231231.csv'

#--------------------------------------------------

df2020 = pd.read_csv(path2020,encoding='euc-kr')
df2023 = pd.read_csv(path2023,encoding='euc-kr')
checker(df2023)
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

  #   Column        Non-Null Count  Dtype 
---  ------        --------------  ----- 
 0   신고년월일         21189 non-null  object
 1   신고시각          21189 non-null  object
 2   발생장소_시        21189 non-null  object
 3   발생장소_구        21189 non-null  object
 4   발생장소_동        21123 non-null  object
 5   발생장소_리        21123 non-null  object
 6   사고원인          21189 non-null  object
 7   사고원인코드명_사고종별  21189 non-null  object
 8   처리결과코드        21189 non-null  object
 9   구조인원          21189 non-null  int64
 행: 21189, 열: 10
'''
# call(df.head(10),'plane')
'''
    신고년월일    신고시각    출동년월일    출동시각    발생장소_시    발생장소_구    발생장소_동    발생장소_리    번지    사고원인    사고원인코드명_사고종별      구조인원
--  ------------  ----------  ------------  ----------  -------------  -------------  -------------  -------------  ------  ----------  -------------------------  ----------
 0  2020-02-01    15:25       2020-02-01    15:29       서울특별시     서대문구       홍제동         홍제동         산1-1   실족추락    실족추락                            1
 1  2020-02-01    14:34       2020-02-01    14:40       서울특별시     종로구         옥인동         옥인동         산3-14  실족추락    실족추락                            1
 2  2020-04-25    10:53       2020-04-25    10:57       서울특별시     종로구         누상동         누상동         산1-27  실족추락    실족추락                            1
 3  2020-07-12    10:55       2020-07-12    10:58       서울특별시     종로구         청운동         청운동         산1-1   개인질환    개인질환                            0
 4  2020-02-19    06:00       2020-02-19    06:06       서울특별시     종로구         청운동         청운동         산4-36  실족추락    실족추락                            0
 5  2020-05-31    12:25       2020-05-31    12:32       서울특별시     종로구         신영동         신영동         82      기타산악    기타산악                            0
 6  2020-04-25    09:35       2020-04-25    09:47       서울특별시     성북구         정릉동         정릉동         산1-1   기타산악    기타산악                            1
 7  2020-11-03    14:02       2020-11-03    14:05       서울특별시     종로구         평창동         평창동         57-1    일반조난    일반조난                            0
 8  2020-06-09    12:55       nan           nan         서울특별시     종로구         구기동         구기동         산3-1   기타산악    기타산악                            0
 9  2020-05-10    14:52       2020-05-10    16:36       서울특별시     종로구         구기동         구기동         산3-1   기타산악    기타산악                            1
'''
df2020.drop('사고원인코드명_사고종별',axis=1,inplace=True)
'''
사고원인 : 산악 1개
사고원인코드명 : 이것저것
처리결과코드 : 233종류

'''

print(df2023.신고년월일.unique())
# call(df2023[df2023.구조인원>=7],'plane')
# call(df2023.head(10),'plane')

print('='*80)
