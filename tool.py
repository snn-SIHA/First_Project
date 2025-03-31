# test

import pandas as pd
import numpy as np
import random

#--------------------------------------------------

# 무작위 난수
a=random.randint(1,6)
print(a)

# 참고용
'''
1 : 소방청 구급 활동정보 #csv
2 : 기상청 생활기상지수 #xlsx <<< 제외
3 : 관광 소비행태 데이터(제주) #xls,csv
4 : 전국 온도(혹은 서울 온도) #csv
5 : 산림청 산불발생통계 #openAPI
6 : 전국 야영장 등록 현황 #csv
'''
print('='*80)

# 데이터프레임 호출 (프로젝트 1 한정)
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