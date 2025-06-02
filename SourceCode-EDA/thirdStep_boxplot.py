import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load the preprocessed Neflix dataset
df = pd.read_csv("netflix_preprocessed_final.csv")


# reconstruct primary genre information from one-hot encoded columns
genre_cols = [col for col in df.columns if col.startswith("genre_")]

#find the gere with maximum value(1) for each row and remove genre prefix
df["primary_genre_grouped"] = df[genre_cols].idxmax(axis=1).str.replace("genre_", "")


# aggregate mean feature values 
director_stats = df.groupby("director").agg({
    "duration_scaled": "mean",
    "rating_encoded": "mean",
    "cast_count": "mean"
}).reset_index()


# extract the most frequent genre
#uses lamda fuction to find the mode of genres 
main_genres = df.groupby("director")["primary_genre_grouped"].agg(lambda x: x.value_counts().idxmax()).reset_index()
director_stats = director_stats.merge(main_genres, on="director")


#Perform K-means clustering on director-level features
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



features = ["duration_scaled", "rating_encoded", "cast_count"]
# Standardize features to have zero mean and unit variance
scaler = StandardScaler()
scaled = scaler.fit_transform(director_stats[features])
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
director_stats["cluster"] = kmeans.fit_predict(scaled)


# Transform data from wide format to long format for visualization
# This reshaping enables grouped boxplot visualization across multiple variables
melted_df = director_stats.melt(id_vars=["cluster"], value_vars=features, var_name="feature", value_name="value")

# Create comparative boxplot visualization showing cluster differences across features
plt.figure(figsize=(12, 6))

#
sns.boxplot(data=melted_df, x="feature", y="value", hue="cluster", palette="Set2")
plt.title("클러스터별 변수 차이 (감독 평균 기준)")
plt.xlabel("변수")
plt.ylabel("값 (정규화됨)")
plt.legend(title="클러스터")
plt.tight_layout()
plt.show()

