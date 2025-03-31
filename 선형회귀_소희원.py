import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from matplotlib import font_manager
from matplotlib import rc
import matplotlib as mpl

mpl.rc('axes', unicode_minus=False)
mpl.rcParams['axes.unicode_minus']=False

font_name = font_manager.FontProperties(fname='c:/windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#--------------------------------------------------------------------------------#

# 1) 데이터 로드
path = './data/seoul/서울refine.csv'
df = pd.read_csv(path, encoding='cp949')

# 2) 계절 정의 (봄, 여름, 가을, 겨울)
seasons = {
    '겨울 (12~2월)': [12, 1, 2],
    '봄 (3~5월)': [3, 4, 5],
    '여름 (6~8월)': [6, 7, 8],
    '가을 (9~11월)': [9, 10, 11]
}

# 3) 계절별 평균 기온 계산 및 Train/Test 분리 적용
seasonal_avg = []
pred_season_temp = {}

for season, months in seasons.items():
    season_data = df[df['월'].isin(months)].groupby('년도')['평균기온(℃)'].mean().reset_index()
    seasonal_avg.append((season, season_data))

    # 4) X(년도), Y(평균 기온) 데이터 분리
    X = season_data['년도'].values.reshape(-1, 1)
    Y = season_data['평균기온(℃)'].values.reshape(-1, 1)

    # 5) Train/Test 데이터 분리 (80 train, 20 test)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 6) 선형 회귀 모델 학습
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # 7) 2025년 예측
    year_2025 = np.array([[2025]])
    pred_temp = model.predict(year_2025)[0][0]
    pred_season_temp[season] = pred_temp

print()
print()

# 8) 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

colors = ['blue', 'green', 'red', 'orange']

for i, (season, season_data) in enumerate(seasonal_avg):
    ax = axes[i]

    # 기존 데이터 그래프(2000~2024)
    ax.plot(season_data['년도'], season_data['평균기온(℃)'], marker='o', linestyle='-', color=colors[i], label='실제 데이터')

    # 2025년 예측값(*)
    ax.scatter(2025, pred_season_temp[season], color='black', marker='*', s=150, label='2025년 예측값')

    # 그래프 세부사항
    ax.set_title(season, fontsize=14, fontweight='bold')
    ax.set_xlabel('년도')
    ax.set_ylabel('평균 기온 (℃)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# 9) 예측 결과 출력
print('2025년 계절별 예상 평균 기온:')
for season, temp in pred_season_temp.items():
    print(f'{season}: {temp:.2f}℃')

print()

