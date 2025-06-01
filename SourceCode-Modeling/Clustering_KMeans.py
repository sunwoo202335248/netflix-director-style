import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 로딩
df = pd.read_csv("netflix_preprocessed_final.csv")

# 2. 장르 복원 (One-Hot → 문자열)
genre_cols = [col for col in df.columns if col.startswith("genre_")]
df["primary_genre_grouped"] = df[genre_cols].idxmax(axis=1).str.replace("genre_", "")

# 3. 감독별 평균 피처 집계
director_stats = df.groupby("director").agg({
    "duration_scaled": "mean",
    "rating_encoded": "mean",
    "cast_count": "mean"
}).reset_index()

# 4. 감독별 대표 장르 추출
main_genres = df.groupby("director")["primary_genre_grouped"].agg(lambda x: x.value_counts().idxmax()).reset_index()
director_stats = director_stats.merge(main_genres, on="director")

# 5. 정규화 (StandardScaler)
features = ["duration_scaled", "rating_encoded", "cast_count"]
scaler = StandardScaler()
scaled = scaler.fit_transform(director_stats[features])

# 6. KMeans 클러스터링
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
director_stats["cluster"] = kmeans.fit_predict(scaled)

# 7. 시각화
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=director_stats,
    x="duration_scaled",
    y="rating_encoded",
    hue="cluster",
    style="primary_genre_grouped",  # 장르별 마커
    palette="Set2",
    alpha=0.7
)
plt.title("🎬 Director Style Clustering (Final Dataset 기반)")
plt.xlabel("Average Scaled Duration")
plt.ylabel("Average Rating Encoded")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
