import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    silhouette_score
)

def evaluate_classification(y_true, y_pred, target_names=None):
    """
    Evaluate classification model performance using accuracy,
    confusion matrix, and classification report. Also visualizes the confusion matrix.

    Args:
        y_true (array-like): Ground truth class labels
        y_pred (array-like): Predicted class labels
        target_names (list): Optional list of class label names for display

    Returns:
        None
    """
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=target_names)

    print(" Accuracy: {:.2f}%".format(acc * 100))
    print("\n Confusion Matrix:\n", cm)
    print("\n Classification Report:\n", report)

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


def evaluate_clustering(X, cluster_labels):
    """
    Calculate the silhouette score to evaluate clustering performance.

    Args:
        X (array-like): Feature matrix used for clustering
        cluster_labels (array-like): Cluster labels assigned to each sample

    Returns:
        float: Silhouette score (ranges from -1 to 1)
    """
    score = silhouette_score(X, cluster_labels)
    print(" Silhouette Score: {:.3f}".format(score))
    return score


#  Example Usage: Test RandomForest classifier on Netflix dataset
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    # 1. Load the preprocessed dataset
    df = pd.read_csv("netflix_preprocessed_final.csv")

    # 2. Recover genre labels from one-hot encoded format
    genre_cols = [col for col in df.columns if col.startswith("genre_")]
    df["primary_genre_grouped"] = df[genre_cols].idxmax(axis=1).str.replace("genre_", "")

    # 3. Encode genre labels into numeric form
    genre_encoder = LabelEncoder()
    df["genre_encoded"] = genre_encoder.fit_transform(df["primary_genre_grouped"])

    # 4. Select input features and labels
    X = df[["duration_scaled", "rating_encoded", "cast_count"]]
    y = df["genre_encoded"]

    # 5. Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # 6. Train a Random Forest classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 7. Run classification evaluation
    evaluate_classification(y_test, y_pred, target_names=genre_encoder.classes_)
