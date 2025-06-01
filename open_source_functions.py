# preprocessing.py
def preprocess_data(csv_path):
    """
    Load and preprocess the Netflix dataset.

    Args:
        csv_path (str): Path to the input CSV file.

    Returns:
        DataFrame: Preprocessed pandas DataFrame.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["director", "duration", "rating", "listed_in"])
    # ... additional processing ...
    return df


# clustering.py
def cluster_directors(df, n_clusters=5):
    """
    Perform k-means clustering on directors based on their average content features.

    Args:
        df (DataFrame): Preprocessed data grouped by director.
        n_clusters (int): Number of clusters.

    Returns:
        tuple: Fitted KMeans model, cluster labels
    """
    from sklearn.cluster import KMeans
    features = df[["duration_scaled", "rating_encoded"]]
    model = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = model.fit_predict(features)
    return model, clusters


# genre_classifier.py
def train_genre_classifier(X, y):
    """
    Train a Decision Tree classifier for predicting genres based on director features.

    Args:
        X (DataFrame): Feature matrix.
        y (Series): Target labels.

    Returns:
        DecisionTreeClassifier: Trained model.
    """
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model


# Evaluation_Metrics.py
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using accuracy and classification report.

    Args:
        model: Trained classification model.
        X_test (DataFrame): Test feature set.
        y_test (Series): True labels for test set.

    Returns:
        dict: Evaluation metrics
    """
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }
