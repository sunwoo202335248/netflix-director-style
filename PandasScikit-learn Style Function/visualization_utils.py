
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_boxplot_and_histogram(df):
    """
    Plot boxplots and histograms for selected numerical features.

    Args:
        df (pd.DataFrame): DataFrame containing at least 'duration_scaled' and 'cast_count'.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.boxplot(data=df, y='duration_scaled', ax=axes[0, 0])
    axes[0, 0].set_title("Boxplot of Duration (Scaled)")

    sns.histplot(data=df, x='duration_scaled', kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("Histogram of Duration (Scaled)")

    sns.boxplot(data=df, y='cast_count', ax=axes[1, 0])
    axes[1, 0].set_title("Boxplot of Cast Count")

    sns.histplot(data=df, x='cast_count', kde=True, ax=axes[1, 1])
    axes[1, 1].set_title("Histogram of Cast Count")

    plt.tight_layout()
    plt.show()

def plot_heatmap(df):
    """
    Plot correlation heatmap for numeric features.

    Args:
        df (pd.DataFrame): DataFrame with numerical features.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def plot_boxplot_by_cluster(df):
    """
    Plot boxplot comparing duration and cast count across clusters.

    Args:
        df (pd.DataFrame): DataFrame with 'duration_scaled', 'cast_count', and 'cluster' columns.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.boxplot(data=df, x='cluster', y='duration_scaled', ax=axes[0])
    axes[0].set_title("Duration by Cluster")

    sns.boxplot(data=df, x='cluster', y='cast_count', ax=axes[1])
    axes[1].set_title("Cast Count by Cluster")

    plt.tight_layout()
    plt.show()

def plot_scatter_by_genre(df):
    """
    Plot scatter plot of cast count vs. duration by genre.

    Args:
        df (pd.DataFrame): DataFrame with 'duration_scaled', 'cast_count', and 'primary_genre_grouped' columns.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='duration_scaled', y='cast_count', hue='primary_genre_grouped')
    plt.title("Scatter Plot of Cast Count vs. Duration by Genre")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Plot confusion matrix heatmap from predictions.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        labels (list): List of label names.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

def plot_director_genre_heatmap(df):
    """
    Plot heatmap showing director and genre relationship.

    Args:
        df (pd.DataFrame): DataFrame with 'director' and genre dummy variables.
    """
    genre_cols = [col for col in df.columns if col.startswith("genre_")]
    pivot = df.groupby("director")[genre_cols].mean()
    top_directors = pivot.sum(axis=1).sort_values(ascending=False).head(20).index
    pivot_top = pivot.loc[top_directors]

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_top, cmap="YlGnBu", annot=True, fmt=".2f")
    plt.title("Top Directors vs. Genre Heatmap")
    plt.tight_layout()
    plt.show()
