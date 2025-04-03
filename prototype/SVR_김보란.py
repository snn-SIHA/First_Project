import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import rc

import pandas as pd
import numpy as np
import seaborn as sns 
import time
import cv2
import folium, json
import geopandas as gpd
import sklearn
from sklearn.preprocessing import MinMaxScaler,StandardScaler # 스케일러
from sklearn.preprocessing import LabelEncoder,OneHotEncoder # 인코더
from sklearn.linear_model import LinearRegression# 모델 : 선형회귀
from sklearn.tree import DecisionTreeClassifier # 모델 : 의사결정트리
from sklearn.ensemble import RandomForestClassifier # 모델 : 랜덤포레스트
from sklearn.svm import SVR # 모델 : SVR
from sklearn.datasets import make_classification # 무작위 데이터 생성기
from sklearn.model_selection import train_test_split # 훈련/평가 데이터 분리
from sklearn.metrics import mean_squared_error, r2_score #평가프로세스
from scipy.stats import skew
from scipy.stats import kurtosis
from sklearn.model_selection import GridSearchCV


# 음수표기 관리
import matplotlib as mpl
mpl.rc('axes', unicode_minus=False)
mpl.rcParams['axes.unicode_minus']=False
pd.set_option("display.unicode.east_asian_width",True) # 유니코드 사용 너비 조정

font_name = font_manager.FontProperties(fname='c:/Windows/Fonts/malgun.ttf').get_name()
mpl.rc('font', family=font_name)
#--------------------------------------------------
'''
문제 정의: 서울지역 월별 기온데이터로 계절별 이상치를 머신러닝에 학습하여 
기후 이상 이벤트를 예측하는 것
작성자 : 김보란
'''
folder='seoul'
filename='서울'
filepath = f'C:/Mtest/project_first/data/{folder}/{filename}.csv'
filesave = f'C:/Mtest/project_first/data/{folder}/{filename}refine.csv'
filetest = f'C:/Mtest/project_first/data/{folder}/{filename}devide.csv'

# 컬럼명을 수동으로 지정해 1~7행 무시
columns = ["연월", "지점", "평균기온(℃)", "평균최저기온(℃)", "평균최고기온(℃)"]
df = pd.read_csv(filepath, encoding="cp949", skiprows=8, skipinitialspace=True, names=columns)
# 결측치 확인 후 처리
print(df[df.isnull().any(axis=1)])
df.drop(index=638, inplace=True)

# strip으로 공백제거
df["연월"] = df["연월"].str.strip()

# 연월 컬럼 연도,월 분리 프로세스
df.연월 = pd.to_datetime(df.연월)
df.insert(1, '연도', df.연월.dt.year)
df.insert(2, '월', df.연월.dt.month)
df.drop('연월',axis=1,inplace=True)

# 왜도 계산
skewness = skew(df['평균기온(℃)'])
# 왜도 그래프
plt.figure(figsize=(12, 6))
sns.distplot(df['평균기온(℃)'], kde=True, color='hotpink', label=f"왜도: {skewness:.2f}")
plt.title("평균기온 분포와 왜도")
plt.xlabel("평균기온(℃)")
plt.ylabel("밀도")
plt.legend()
plt.tight_layout()
plt.show()

# plt.figure(figsize=(12, 6))
# for col in df_3_columns:
#     sns.histplot(df[col], kde=True, label=f"{col} (왜도: {skew(df[col]):.4f})", bins=20, alpha=0.7)
# plt.title("데이터의 분포와 왜도")
# plt.xlabel("값")
# plt.ylabel("밀도")
# plt.legend()
# plt.tight_layout()
# plt.show()
# sns.distplot(df['평균기온(℃)'], color='hotpink', label=df['평균기온(℃)'].skew())

# dftest : 2024~2025
dftest = df[(df.연도==2024) | (df.연도==2025)]
dftest.reset_index(drop=True, inplace=True)
print()
print(dftest.head(3), '\n', dftest.tail(3))
print()
# 지점(col) 제거
dftest.drop(columns='지점', inplace=True)

# df : 2000~2023
df = df[(df.연도>=2000)&(df.연도<=2023)]
df.reset_index(drop=True, inplace=True)
# 지점(col) 제거
df.drop(columns='지점', inplace=True)

# 데이터 저장(코드 상에서만 확인, 실제 경로 저장은 주석으로)
# df.to_csv(filesave, index=False)
# dftest.to_csv(filetest, index=False)

# z-score로 이상치 확인 
mean_mean_temp = df['평균기온(℃)'].mean()
std_mean_temp = df['평균기온(℃)'].std()
df['z_score'] = (df['평균기온(℃)'] - mean_mean_temp ) / std_mean_temp
threshold = 2
df['outliers'] = np.abs(df['z_score']) > threshold

# 계절 칼럼 추가
def get_season(month):
    if month in [3, 4, 5]:
        return '봄'
    elif month in [6, 7, 8]:
        return '여름'
    elif month in [9, 10, 11]:
        return '가을'
    else:
        return '겨울'

df['계절'] = df['월'].apply(get_season)

#------------------------- 스케일링/선형회귀 학습 ------------------------

# 1) 특성과 타겟 분리
X = df.loc[:, '연도':'월']     # '연도'부터 '월'까지를 특성으로 사용
y = df['평균기온(℃)']

# 2) 훈련/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3) 스케일링
scaler = MinMaxScaler()
# 학습데이터 스케일 fit + transform
X_train_scaled = scaler.fit_transform(X_train)
# 평가데이터 스케일
X_test_scaled = scaler.fit_transform(X_test)

# 4) lr: LinearRegression 모델 학습
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# 5) lr: 예측 및 성능 확인
y_pred = lr.predict(X_test_scaled)

print("===== 학습 데이터 분할 후 기본 성능 평가 =====")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f" > MSE: {mse:.4f}")
print(f" > RMSE: {mse**0.5:.4f}")
print(f" > R2: {r2:.4f}\n")

'''
 > MSE: 100.7674
 > RMSE: 10.0383
 > R2: 0.1388
'''

#-------------------------LR: dftest 예측 ------------------------

# dftest의 실제 X, y 준비
X_dftest = dftest.loc[:, '연도':'월']
y_dftest = dftest['평균기온(℃)']

# dftest도 위에서 fit한 스케일러로 transform만 수행
X_dftest_scaled = scaler.transform(X_dftest)

# lr: 예측
y_pred_dftest_lr = lr.predict(X_dftest_scaled)

# lr: 예측 결과와 실제값 비교
print("===== 2024~2025.2월까지 dftest 예측 결과 비교 =====")
compare_df = pd.DataFrame({
    '연도': X_dftest['연도'],
    '월': X_dftest['월'],
    '실제평균기온(℃)': y_dftest,
    '예측평균기온(℃)': y_pred_dftest_lr
})

# 간단히 앞부분만 확인
print(compare_df.head(10))

# lr: 성능 지표 (2024~2025년 예측)
mse_dftest = mean_squared_error(y_dftest, y_pred_dftest_lr)
r2_dftest = r2_score(y_dftest, y_pred_dftest_lr)
print(f"\n===== LR_minmax: dftest 성능 지표(2024~2025) =====")
print(f" > MSE: {mse_dftest:.4f}")
print(f" > RMSE: {mse_dftest**0.5:.4f}")
print(f" > R2: {r2_dftest:.4f}")
'''
===== LR_minmax : dftest 성능 지표(2024~2025) =====
 > MSE: 100.7674
 > RMSE: 10.0383
 > R2: 0.1388
'''
# 1) 2025년의 월(1~12) 가진 데이터프레임 생성
predict_2025 = pd.DataFrame({'연도':[2025]*12, '월':range(1,13)})

# 2) 위에서 사용한 scaler로 transform
predict_2025_scaled = scaler.transform(predict_2025)

# 3) LR: 예측
pred_2025_lr = lr.predict(predict_2025_scaled)

# 4) 결과 정리
result_2025_lr = pd.DataFrame({
    '연도': predict_2025['연도'],
    '월': predict_2025['월'],
    '2025년 예측평균기온(℃)': pred_2025_lr
})

print("===== lr 2025년 1월~12월 월별 평균기온 예측 =====")
print(result_2025_lr)

print("===== LR: 기본 성능 평가(테스트셋) =====")
print(f" > MSE: {mse:.4f}")
print(f" > RMSE: {mse**0.5:.4f}")
print(f" > R2: {r2:.4f}\n")

'''
 > MSE: 95.7627
 > RMSE: 9.7858
 > R2: -0.0222
'''

compare_df.to_csv('C:/Mtest/project_first/data/seoul/LR예측결과_2024_2025_minMax.csv', index=False)

#------------------------- SVR 모델 학습 및 예측 ------------------------
X_dftest = dftest[['연도','월']]
y_dftest = dftest['평균기온(℃)']

# 스케일 변환
X_dftest_scaled = scaler.transform(X_dftest)


# svr: svm기반 회귀모델 학습
svr = SVR(kernel='rbf', C=3, epsilon=0.1)
svr.fit(X_train_scaled, y_train)

# 예측
y_pred_dftest_svr = svr.predict(X_dftest_scaled)

print("===== SVR 2024~2025.2월까지 dftest 예측 결과 비교 =====")
compare_df_svr = pd.DataFrame({
    '연도': X_dftest['연도'],
    '월': X_dftest['월'],
    '실제평균기온(℃)': y_dftest,
    '예측평균기온(℃)': y_pred_dftest_svr
})
print(compare_df_svr.head(12))  # 일부만 출력

# SVR 성능 지표
mse_dftest = mean_squared_error(y_dftest, y_pred_dftest_svr)
r2_dftest = r2_score(y_dftest, y_pred_dftest_svr)
print(f"\n===== SVR: dftest(2024~2025.2월) 성능 지표 =====")
print(f" > MSE: {mse_dftest:.4f}")
print(f" > RMSE: {mse_dftest**0.5:.4f}")
print(f" > R2: {r2_dftest:.4f}\n")
'''
===== SVR: dftest(2024~2025.2월) 성능 지표 =====
 > MSE: 12.2232
 > RMSE: 3.4962
 > R2: 0.8955

 > MSE: 9.3641
 > RMSE: 3.0601
 > R2: 0.9200
'''
# ------------------- svr: 2025년 1~12월 월별 예측 -------------------
predict_2025 = pd.DataFrame({'연도':[2025]*12, '월':range(1,13)})
pred_2025_svr = svr.predict(predict_2025_scaled)

# 4) 결과 정리
result_2025_svr = pd.DataFrame({
    '연도': predict_2025['연도'],
    '월': predict_2025['월'],
    '2025년 예측평균기온(℃)': pred_2025_svr
})

new_row = pd.DataFrame([{
    '연도': 2025, 
    '월' : '연평균',
    '2025년 예측평균기온(℃)': pred_2025_svr.mean()
}])

result_2025_svr= pd.concat([result_2025_svr, new_row], ignore_index=True)


print('===== svr 2025년 1월~12월 월별 평균기온 예측 =====')
print(result_2025_svr)

result_2025_svr.to_csv('C:/Mtest/project_first/data/seoul/SVR_예측결과_2024_2025_c=3.csv', index=False)

'''
감마(gamma) 값은 RBF(Radial Basis Function) 커널을 사용하는 SVR(또는 SVC)에서 매우 중요한 하이퍼파라미터 중 하나입니다.

gamma가 클수록 RBF 커널의 폭(반경)이 좁아져, 지역적인 부분을 더 세밀하게(날카롭게) 학습합니다.

이는 훈련 데이터에 더 민감하게 반응하기 때문에 복잡한 결정 경계를 만들 수 있고, 결과적으로 R²가 올라가는 경우가 많습니다.

단, gamma가 너무 크면 오버피팅(과적합) 위험도 함께 증가할 수 있습니다.

즉, 감마를 1 → 2로 높였을 때 R²가 개선된 것은, 모델이 더 복잡해지면서 기존 데이터 패턴을 조금 더 정교하게 맞추는 쪽으로 동작했기 때문입니다.
이때, **다른 지표(예: 검증 세트나 교차검증 점수)**에서도 성능이 지속적으로 좋아지는지 확인해보면, 과적합 없이 실질적인 개선이 이루어졌는지 판단할 수 있습니다.
'''

'''
다음 단계 권장 사항
GridSearchCV / RandomizedSearchCV

단순히 gamma=2로 바꿨을 때 좋아졌다 하더라도, 다른 C, epsilon, gamma 조합이 더 좋을 수도 있습니다.

가능한 후보군을 설정한 뒤, 교차검증을 통해 가장 적합한 파라미터를 찾는 것이 좋습니다.

교차검증 점수(또는 별도의 검증 셋 성능) 확인

훈련 데이터에 대한 R²만 확인하지 말고, 검증 데이터나 교차검증으로 얻은 평균 성능도 살펴보세요.

만약 훈련 점수는 계속 좋아지는데 검증 점수가 나빠진다면, 오버피팅을 의심해야 합니다.

더 많은 특성(Feature) 활용

같은 모델이라도 기온 예측에 유의미한 **추가 특성(예: 전월 기온, 강수량, 습도 등)**을 확보하면, 성능 향상의 여지가 큽니다.
'''

'''감마가 아니라 c였다!
C를 키웠더니 성능(R²)이 좋아졌다면, 이는 오차를 더 강하게 억제함으로써 모델이 훈련 데이터에 더 치밀하게 맞춘 결과입니다.

이것이 실제로도 일반화 성능 향상으로 이어지는지, 아니면 과적합인지를 확인하기 위해선 교차검증과 테스트 성능을 비교해야 합니다.

교차검증에서 안정적으로 높은 점수를 얻고, 테스트셋 예측력도 향상되었다면, C를 높인 것이 유효한 것으로 결론지을 수 있습니다.

추가로, GridSearchCV 등을 통해 C, epsilon, gamma를 체계적으로 탐색하면, 최적의 조합을 빠르게 찾아볼 수 있습니다.
'''





# 4월, 6,7,8,9월의 오차가 큼.
'''
    연도  월  실제평균기온(℃)  예측평균기온(℃)
0   2024   1             -0.5         1.024693
1   2024   2              3.8         3.923759
2   2024   3              7.0         8.173821
3   2024   4             16.3        13.238865
4   2024   5             18.5        18.275058
5   2024   6             24.6        22.264447
6   2024   7             26.6        24.233647
7   2024   8             29.3        23.544838
8   2024   9             25.5        20.173358
9   2024  10             16.7        14.821237
10  2024  11              9.7         8.748390
11  2024  12              0.8         3.349549
'''


