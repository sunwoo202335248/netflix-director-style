import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load preprocessed dataset
df = pd.read_csv("netflix_preprocessed_final.csv")

# 2. Restore genre labels from one-hot encoded columns (e.g., genre_Drama → Drama)
genre_cols = [col for col in df.columns if col.startswith("genre_")]
df["primary_genre_grouped"] = df[genre_cols].idxmax(axis=1).str.replace("genre_", "")

# 3. Aggregate average features for each director
# - Average duration, rating, and number of cast members per director
director_stats = df.groupby("director").agg({
    "duration_scaled": "mean",
    "rating_encoded": "mean",
    "cast_count": "mean"
}).reset_index()

# 4. Determine the most common genre for each director
main_genres = df.groupby("director")["primary_genre_grouped"].agg(
    lambda x: x.value_counts().idxmax()
).reset_index()
director_stats = director_stats.merge(main_genres, on="director")

# 5. Normalize features for clustering
features = ["duration_scaled", "rating_encoded", "cast_count"]
scaler = StandardScaler()
scaled = scaler.fit_transform(director_stats[features])

# 6. Perform KMeans clustering
# - Group directors into clusters based on their average content features
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
director_stats["cluster"] = kmeans.fit_predict(scaled)

# 7. Visualize the clustering result
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=director_stats,
    x="duration_scaled",
    y="rating_encoded",
    hue="cluster",                         # Cluster group coloring
    style="primary_genre_grouped",        # Marker style by main genre
    palette="Set2",
    alpha=0.7
)
plt.title("🎬 Director Style Clustering (Based on Final Dataset)")
plt.xlabel("Average Scaled Duration")
plt.ylabel("Average Rating Encoded")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
