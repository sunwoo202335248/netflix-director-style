
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.cluster import KMeans

def load_data(filepath="netflix_preprocessed_final.csv"):
    return pd.read_csv(filepath)

def visualize_distribution(df):
    for var in ["duration_scaled", "rating_encoded", "cast_count"]:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        sns.boxplot(y=df[var])
        plt.title(f'Boxplot of {var}')
        plt.subplot(1, 2, 2)
        sns.histplot(df[var], kde=True)
        plt.title(f'Distribution of {var}')
        plt.tight_layout()
        plt.show()

def plot_correlation_heatmap(df):
    corr = df[["duration_scaled", "rating_encoded", "cast_count"]].corr()
    plt.figure(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def cluster_boxplot_by_director(df):
    genre_cols = [col for col in df.columns if col.startswith("genre_")]
    df["primary_genre_grouped"] = df[genre_cols].idxmax(axis=1).str.replace("genre_", "")
    director_stats = df.groupby("director")[["duration_scaled", "rating_encoded", "cast_count"]].mean().reset_index()
    main_genres = df.groupby("director")["primary_genre_grouped"].agg(lambda x: x.value_counts().idxmax()).reset_index()
    director_stats = director_stats.merge(main_genres, on="director")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(director_stats[["duration_scaled", "rating_encoded", "cast_count"]])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    director_stats["cluster"] = kmeans.fit_predict(scaled)

    melted = director_stats.melt(id_vars=["cluster"], value_vars=["duration_scaled", "rating_encoded", "cast_count"],
                                 var_name="feature", value_name="value")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted, x="feature", y="value", hue="cluster", palette="Set2")
    plt.title("Cluster Comparison by Feature")
    plt.tight_layout()
    plt.show()

def compare_model_confusion_matrix(df):
    genre_cols = [col for col in df.columns if col.startswith("genre_")]
    df["primary_genre_grouped"] = df[genre_cols].idxmax(axis=1).str.replace("genre_", "")
    genre_encoder = LabelEncoder()
    df["genre_encoded"] = genre_encoder.fit_transform(df["primary_genre_grouped"])
    X = df[["duration_scaled", "rating_encoded", "cast_count"]]
    y = df["genre_encoded"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    rf = RandomForestClassifier(random_state=42, class_weight="balanced", min_samples_split=5)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, display_labels=genre_encoder.classes_,
                                            cmap="Blues", ax=axes[0], xticks_rotation=45)
    axes[0].set_title("Random Forest")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_dt, display_labels=genre_encoder.classes_,
                                            cmap="Oranges", ax=axes[1], xticks_rotation=45)
    axes[1].set_title("Decision Tree")
    plt.tight_layout()
    plt.show()

    print(f" Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)*100:.2f}%")
    print(f" Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt)*100:.2f}%")

def visualize_director_genre_heatmap(df):
    genre_cols = [col for col in df.columns if col.startswith("genre_")]
    df["primary_genre_grouped"] = df[genre_cols].idxmax(axis=1).str.replace("genre_", "")
    genre_encoder = LabelEncoder()
    df["genre_encoded"] = genre_encoder.fit_transform(df["primary_genre_grouped"])
    X = df[["duration_scaled", "rating_encoded", "cast_count"]]
    y = df["genre_encoded"]

    clf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    clf.fit(X, y)
    pred = clf.predict(X)
    df["predicted_genre"] = genre_encoder.inverse_transform(pred)

    genre_dist = df.groupby("director")["predicted_genre"].value_counts(normalize=True).unstack().fillna(0)
    top_directors = df["director"].value_counts().head(10).index
    top_data = genre_dist.loc[top_directors]

    plt.figure(figsize=(12, 6))
    sns.heatmap(top_data, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Top Directors' Predicted Genre Distribution")
    plt.tight_layout()
    plt.show()

def visualize_director_scatterplot(df):
    genre_cols = [col for col in df.columns if col.startswith("genre_")]
    df["primary_genre_grouped"] = df[genre_cols].idxmax(axis=1).str.replace("genre_", "")

    director_stats = df.groupby("director").agg({
        "duration_scaled": "mean",
        "rating_encoded": "mean",
        "cast_count": "mean"
    }).reset_index()

    main_genres = df.groupby("director")["primary_genre_grouped"].agg(lambda x: x.value_counts().idxmax()).reset_index()
    director_stats = director_stats.merge(main_genres, on="director")

    features = ["duration_scaled", "rating_encoded", "cast_count"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(director_stats[features])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    director_stats["cluster"] = kmeans.fit_predict(scaled)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=director_stats,
        x="duration_scaled",
        y="rating_encoded",
        hue="cluster",
        style="primary_genre_grouped",
        palette="Set2",
        alpha=0.7
    )
    plt.title(" Director Style Clustering")
    plt.xlabel("Average Scaled Duration")
    plt.ylabel("Average Rating Encoded")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_data()
    visualize_distribution(df)
    plot_correlation_heatmap(df)
    cluster_boxplot_by_director(df)
    visualize_director_scatterplot(df)
    compare_model_confusion_matrix(df)
    visualize_director_genre_heatmap(df)
