# csv_devider
'''
본 코드 동작 방식

기상청 csv 데이터 대상으로 2000년~2024년 데이터 추출 후 .refine으로 저장
같은 데이터 대상으로 2025년 데이터 추출 후 .devide로 저장
'''
import pandas as pd
import numpy as np
pd.set_option("display.max_colwidth",20) # 출력할 열의 너비
pd.set_option("display.unicode.east_asian_width",True) # 유니코드 사용 너비 조정

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

folder   = 'ulsan' # 폴더명 입력
filename = '울산'  # 파일명 입력
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
print(data.head())
print(data.tail())
print('- '*40)

print(testdata)
#data = data[~data["년월"].str.contains("2025", na=False)] # 방안 2

# data["년월"] = pd.to_datetime(data["년월"], format="%Y-%m") # 방안 3
# data["연도"] = data["년월"].dt.year # 연도만 추출
# data = data[data["연도"] != 2025]  # 2025년 데이터 제거
#print(data.head())

# 아래는 저장 기능을 담당하는 코드
# data.to_csv(pathsave,index=False,encoding='cp949')
# testdata.to_csv(pathtest,index=False,encoding='cp949')

print('='*80)
