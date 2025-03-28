import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc

# 한글 폰트 설정
font_name = font_manager.FontProperties(fname="c:/windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

# CSV 파일 읽기
data = pd.read_csv("./data/seoul/서울refine.csv", encoding="cp949")

# 열 이름 명시적으로 설정
data.columns = ['연도', '월', '지점', '평균기온', '최저기온', '최고기온']

# 연도별 평균 기온 변화 시각화
plt.figure(figsize=(15, 6))
yearly_avg_temp = data.groupby('연도')['평균기온'].mean()

plt.plot(yearly_avg_temp.index, yearly_avg_temp.values, marker='o')
plt.title('연도별 평균 기온 변화', fontsize=15)
plt.xlabel('연도', fontsize=12)
plt.ylabel('평균 기온 (°C)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print(f'그래프 1 출력 완료\n{'- '*40}')

# 기온 비교 시각화
plt.figure(figsize=(15, 6))
seasonal_data = data.groupby(['연도', '월'])[['평균기온', '최저기온', '최고기온']].mean().reset_index()

plt.plot(seasonal_data['연도'] + seasonal_data['월']/12, 
         seasonal_data['평균기온'], 
         label='평균 기온', 
         marker='o', 
         markersize=3)
plt.plot(seasonal_data['연도'] + seasonal_data['월']/12, 
         seasonal_data['최저기온'], 
         label='최저 기온', 
         marker='s', 
         markersize=3)
plt.plot(seasonal_data['연도'] + seasonal_data['월']/12, 
         seasonal_data['최고기온'], 
         label='최고 기온', 
         marker='^', 
         markersize=3)

plt.title('평균, 최저, 최고 기온 변화', fontsize=15)
plt.xlabel('연도', fontsize=12)
plt.ylabel('기온 (°C)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print(f'그래프 2 출력 완료\n{'- '*40}')

# 이상 기온 탐색 (박스플롯)
plt.figure(figsize=(15, 6))
sns.boxplot(x='연도', y='평균기온', data=data)
plt.title('연도별 평균 기온 이상치 탐색', fontsize=15)
plt.xlabel('연도', fontsize=12)
plt.ylabel('평균 기온 (°C)', fontsize=12)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

print(f'그래프 3 출력 완료\n{'- '*40}')

# 추가: 이상치 통계 출력
def find_outliers(group):
    Q1 = group.quantile(0.25)
    Q3 = group.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return group[(group < lower_bound) | (group > upper_bound)]

yearly_outliers = data.groupby('연도')['평균기온'].apply(find_outliers)
print("이상치 연도 및 값:")
print(yearly_outliers)

'''
Series([], Name: 평균기온, dtype: float64)

-GPT-
이 출력은 "평균기온"이라는 이름을 가진 Series가 있지만,
값이 비어 있다는 뜻이에요. 즉, Series는 생성됐지만,
데이터가 없다는 뜻이죠.
'''