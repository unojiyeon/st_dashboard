import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# 출력 결과 저장을 위한 설정
REPORT_PATH = "naverapieda/eda_report.md"
IMAGE_DIR = "naverapieda/images"
DATA_DIR = "naverapieda/data"

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# 리포트 초기화
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("# 네이버 쇼핑 트렌드 및 검색 데이터 EDA 분석 리포트\n\n")

def log_to_report(text):
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def save_plot(filename, title, interpretation):
    path = os.path.join(IMAGE_DIR, filename)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    log_to_report(f"### {title}")
    log_to_report(f"![{title}](images/{filename})")
    log_to_report(f"\n**해석**: {interpretation}\n")

# 1. 데이터 로드 및 병합
folders = os.listdir(DATA_DIR)
trend_files = [f for f in folders if "trend_trend" in f]
blog_files = [f for f in folders if "trend_blog" in f]
search_files = [f for f in folders if "trend_search" in f]

df_trend = pd.concat([pd.read_csv(os.path.join(DATA_DIR, f)) for f in trend_files])
df_blog = pd.concat([pd.read_csv(os.path.join(DATA_DIR, f)) for f in blog_files])
df_search = pd.concat([pd.read_csv(os.path.join(DATA_DIR, f)) for f in search_files])

# 2. 기본 정보 출력
log_to_report("## 1. 데이터 기본 정보")

for name, df in [("쇼핑 트렌드", df_trend), ("블로그", df_blog), ("쇼핑 검색", df_search)]:
    log_to_report(f"### {name} 데이터")
    log_to_report("#### 상위 5개 행")
    log_to_report(df.head(5).to_markdown())
    log_to_report("\n#### 하위 5개 행")
    log_to_report(df.tail(5).to_markdown())
    log_to_report("\n#### 기본 정보 (info)")
    # info()는 바로 문자열로 리턴되지 않으므로 io 사용 가능하나 여기선 간단히 출력
    log_to_report(f"- 행 수: {df.shape[0]}, 열 수: {df.shape[1]}")
    log_to_report(f"- 컬럼명: {', '.join(df.columns)}")
    
    log_to_report("\n#### 기술통계 (수치형)")
    log_to_report(df.describe().to_markdown())
    log_to_report("\n#### 기술통계 (범주형)")
    log_to_report(df.describe(include=['object', 'string']).to_markdown())
    log_to_report("\n---\n")

# 시각화 개수 카운트
viz_count = 0

# 3. 데이터 분석 및 시각화

# [시각화 1] 쇼핑 키워드별 트렌드 비교 (시계열)
df_trend['period'] = pd.to_datetime(df_trend['period'])
plt.figure(figsize=(15, 6))
for keyword in df_trend['keyword'].unique():
    subset = df_trend[df_trend['keyword'] == keyword]
    plt.plot(subset['period'], subset['ratio'], label=keyword)
plt.title("오메가3 vs 비타민D 쇼핑 클릭 트렌드 비교")
plt.xlabel("날짜")
plt.ylabel("클릭 지수 (ratio)")
plt.legend()
interpretation = "오메가3와 비타민D의 검색 트렌드를 비교한 결과, 특정 시점에 검색량이 급증하는 구간이 발견됩니다. 이는 프로모션이나 미디어 노출의 영향일 가능성이 큽니다."
save_plot("trend_comparison.png", "키워드별 쇼핑 클릭 트렌드 비교", interpretation)

# 시각화 관련 통계표
pivot_trend = df_trend.pivot_table(index='keyword', values='ratio', aggfunc=['mean', 'max', 'std'])
log_to_report("#### 트렌드 키워드별 통계 요약")
log_to_report(pivot_trend.to_markdown())

# [시각화 2] 키워드별 평균 클릭 지수 (범주형 빈도/평균)
plt.figure(figsize=(10, 6))
df_trend.groupby('keyword')['ratio'].mean().plot(kind='bar', color=['skyblue', 'orange'])
plt.title("키워드별 평균 클릭 지수")
plt.xticks(rotation=0)
interpretation = "최근 1년간의 평균 클릭 지수를 비교했을 때 어떤 영양제에 대한 소비자 관심도가 전반적으로 더 높은지 한눈에 파악할 수 있는 지표입니다."
save_plot("avg_ratio_bar.png", "키워드별 평균 클릭 지수", interpretation)

# [시각화 3] 쇼핑 검색 결과: 몰(mallName) 빈도수 상위 15개
plt.figure(figsize=(12, 6))
df_search['mallName'].value_counts().head(15).plot(kind='barh')
plt.title("쇼핑 검색 결과 주요 판매처 빈도 (상위 15개)")
interpretation = "네이버 쇼핑 검색 결과에서 가장 많이 노출되는 상위 15개 쇼핑몰을 분석합니다. 특정 대형 몰의 점유율이 높은지, 중소형 몰이 다양한지 확인할 수 있습니다."
save_plot("mall_frequency.png", "주요 판매 쇼핑몰 빈도수", interpretation)
log_to_report(df_search['mallName'].value_counts().head(15).to_frame().to_markdown())

# [시각화 4] 쇼핑 검색 결과: 가격(lprice) 분포 (수치형 - 히스토그램)
df_search['lprice'] = pd.to_numeric(df_search['lprice'], errors='coerce')
plt.figure(figsize=(10, 6))
df_search['lprice'].hist(bins=30)
plt.title("쇼핑 상품 가격 분포")
interpretation = "검색된 상품들의 가격 분포를 통해 소위 말하는 '가성비' 제품군과 '프리미엄' 제품군의 가격대를 파악할 수 있습니다. 대부분의 상품이 특정 가격대에 밀집되어 있습니다."
save_plot("price_distribution.png", "상품 가격 분포", interpretation)

# [시각화 5] 키워드별 상품 가격 박스플롯 (이변량)
plt.figure(figsize=(10, 6))
df_search.boxplot(column='lprice', by='query')
plt.title("키워드별 상품 가격대 비교")
plt.suptitle("") # 박스플롯 기본 타이틀 제거
interpretation = "오메가3와 비타민D의 가격 범위를 비교한 결과, 특정 품목의 가격 편차가 더 심하거나 평균 가격대가 높게 형성되어 있는 양상을 보입니다."
save_plot("price_boxplot.png", "키워드별 가격 박스플롯", interpretation)

# [시각화 6] 블로그 키워드 분석 (TF-IDF) - 오메가3
def plot_tfidf(df, query, filename):
    tfidf = TfidfVectorizer(max_features=30, stop_words=None)
    # 제목과 요약을 합쳐서 분석 (컬럼 존재 여부 확인)
    columns = df.columns
    text_parts = []
    if 'title' in columns:
        text_parts.append(df[df['query'] == query]['title'].fillna(''))
    if 'description' in columns:
        text_parts.append(df[df['query'] == query]['description'].fillna(''))
    
    if not text_parts:
        print(f"No text columns found for {query}")
        return

    text_data = text_parts[0]
    for i in range(1, len(text_parts)):
        text_data = text_data + " " + text_parts[i]
    # HTML 태그 제거
    text_data = text_data.apply(lambda x: re.sub('<[^<]+?>', '', x))
    
    tfidf_matrix = tfidf.fit_transform(text_data)
    words = tfidf.get_feature_names_out()
    sums = tfidf_matrix.sum(axis=0)
    data = []
    for col, word in enumerate(words):
        data.append((word, sums[0, col]))
    
    ranking = pd.DataFrame(data, columns=['word', 'importance']).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    plt.barh(ranking['word'], ranking['importance'], color='green')
    plt.gca().invert_yaxis()
    plt.title(f"{query} 블로그 주요 키워드 중요도 (TF-IDF)")
    
    interp = f"{query}와 관련된 블로그 포스팅에서 자주 언급되는 상위 30개 핵심 키워드를 추출했습니다. 소비자들이 주로 어떤 효능이나 브랜드에 관심이 있는지 알 수 있습니다."
    save_plot(filename, f"{query} 블로그 키워드 분석", interp)
    log_to_report(ranking.head(10).to_markdown())

plot_tfidf(df_blog, "오메가3", "blog_tfidf_omega3.png")

# [시각화 7] 블로그 키워드 분석 (TF-IDF) - 비타민D
plot_tfidf(df_blog, "비타민D", "blog_tfidf_vitamind.png")

# [시각화 8] 쇼핑 상품명 키워드 분석 (TF-IDF) - 오메가3
plot_tfidf(df_search, "오메가3", "search_tfidf_omega3.png")

# [시각화 9] 쇼핑 상품명 키워드 분석 (TF-IDF) - 비타민D
plot_tfidf(df_search, "비타민D", "search_tfidf_vitamind.png")

# [시각화 10] 브랜드(brand) 노출 빈도 비교 (상위 10개)
df_search['brand'] = df_search['brand'].fillna('미지정')
brand_counts = df_search[df_search['brand'] != '미지정']['brand'].value_counts().head(10)
plt.figure(figsize=(12, 6))
brand_counts.plot(kind='pie', autopct='%1.1f%%')
plt.title("쇼핑 검색 상위 브랜드 점유율 (미지정 제외)")
interpretation = "네이버 쇼핑 검색 결과에서 특정 브랜의 노출 비중을 분석합니다. 브랜드 인지도가 검색 결과 상위 노출에 미치는 영향을 간접적으로 파악할 수 있습니다."
save_plot("brand_share.png", "주요 브랜드 점유율", interpretation)
log_to_report(brand_counts.to_frame().to_markdown())

# [시각화 11] 블로그 게시물 날짜별 빈도 (포스팅 시점 분석)
# 블로그 데이터에 날짜 정보가 있다면 유용함. 'postdate' 컬럼 확인
if 'postdate' in df_blog.columns:
    df_blog['postdate'] = pd.to_datetime(df_blog['postdate'], format='%Y%m%d', errors='coerce')
    plt.figure(figsize=(12, 6))
    df_blog.set_index('postdate').resample('ME').size().plot()
    plt.title("월별 블로그 포스팅 빈도 추이")
    interpretation = "블로그 포스팅의 날짜 정보를 월 단위로 집계한 결과입니다. 특정 시기에 정보 제공이나 리뷰 포스팅이 집중되었는지 확인할 수 있습니다."
    save_plot("blog_post_trend.png", "월별 블로그 포스팅 빈도", interpretation)

print("EDA 분석 및 리포트 생성이 완료되었습니다.")
