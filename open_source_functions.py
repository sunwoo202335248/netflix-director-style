# preprocessing.py
def preprocess_netflix_data(csv_path):
    """
    Preprocess the raw Netflix CSV file and return a cleaned DataFrame.

    Args:
        csv_path (str): Path to the original CSV file (e.g., netflix_titles.csv)

    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    df = pd.read_csv(csv_path)

    # Select columns to use
    columns = [
        'show_id', 'type', 'title', 'director', 'cast', 'country',
        'date_added', 'release_year', 'rating', 'duration', 'listed_in', 'description'
    ]
    df = df[columns]

    # 1. Remove rows with missing key values
    df = df.dropna(subset=['director', 'rating', 'duration', 'listed_in'])

    # 2. Convert duration to numeric
    def extract_duration(value):
        try:
            return int(value.strip().split(' ')[0])
        except:
            return None
    df['duration_number'] = df['duration'].apply(extract_duration)
    df = df.dropna(subset=['duration_number'])

    # 3. Genre mapping function
    def map_genre(genre):
        genre = genre.lower()
        if 'drama' in genre or 'romantic' in genre:
            return 'Drama'
        elif 'comedy' in genre:
            return 'Comedy'
        elif 'documentary' in genre:
            return 'Documentary'
        elif 'action' in genre or 'thriller' in genre or 'horror' in genre:
            return 'Action'
        elif 'children' in genre or 'kids' in genre or 'family' in genre:
            return 'Children'
        else:
            return 'Other'

    # 4. Group primary genre based on first listed_in label
    df['primary_genre_grouped'] = df['listed_in'].apply(lambda x: map_genre(x.split(',')[0]))

    # 5. Feature: count of cast members
    df['cast_count'] = df['cast'].apply(lambda x: len(x.split(',')) if pd.notnull(x) else 0)

    # 6. Encode rating
    rating_encoder = LabelEncoder()
    df['rating_encoded'] = rating_encoder.fit_transform(df['rating'])

    # 7. One-Hot Encoding for genre and type
    df = pd.get_dummies(df, columns=['primary_genre_grouped', 'type'], prefix=['genre', 'type'])

    # 8. Normalize duration
    scaler = StandardScaler()
    df['duration_scaled'] = scaler.fit_transform(df[['duration_number']])

    return df


# Clustering_KMeans.py
def cluster_directors(df, n_clusters=4):
    """
    Perform KMeans clustering on directors based on their average content features.

    Args:
        df (pd.DataFrame): Preprocessed Netflix dataset
        n_clusters (int): Number of clusters

    Returns:
        pd.DataFrame: Director-level DataFrame with cluster labels
    """
    from sklearn.cluster import KMeans
    # clustering logic omitted
    return df


# Classification_DecisionTree.py
def train_decision_tree(X, y, max_depth=5):
    """
    Train a Decision Tree model to classify genres based on content features.

    Args:
        X (pd.DataFrame): Input features
        y (pd.Series): Genre labels
        max_depth (int): Maximum depth of the tree

    Returns:
        DecisionTreeClassifier: Trained model
    """
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X, y)
    return model


# Random_Forest.py
def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier using content-level features.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels

    Returns:
        RandomForestClassifier: Trained model
    """
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_split=5,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf


# GridSearch_Tuning.py
def tune_hyperparameters(X_train, y_train, param_grid):
    """
    Perform hyperparameter tuning for Random Forest using GridSearchCV.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        param_grid (dict): Dictionary of parameters to search

    Returns:
        GridSearchCV: Fitted GridSearch object with best parameters
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(RandomForestClassifier(random_state=42),
                        param_grid=param_grid, cv=5, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid


# cross_validation.py
def run_cross_validation(X, y, cv=10):
    """
    Perform cross-validation to evaluate generalization performance (F1-macro).

    Args:
        X (pd.DataFrame): Input features
        y (pd.Series): Target labels
        cv (int): Number of folds

    Returns:
        list: F1 scores for each fold
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import make_scorer, f1_score
    from sklearn.model_selection import cross_val_score
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    f1_macro = make_scorer(f1_score, average='macro')
    scores = cross_val_score(model, X, y, cv=cv, scoring=f1_macro)
    return scores


# Evaluation_Metrics.py
def evaluate_classification(y_true, y_pred, target_names=None):
    """
    Print and visualize classification metrics such as accuracy, confusion matrix, and report.
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import seaborn as sns
    import matplotlib.pyplot as plt

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")
    print("Confusion Matrix: ", cm)
    print("Classification Report: ", classification_report(y_true, y_pred, target_names=target_names))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def evaluate_clustering(X, cluster_labels):
    """
    Calculate the silhouette score to evaluate clustering quality.

    Args:
        X (pd.DataFrame): Feature data
        cluster_labels (array): Cluster assignments

    Returns:
        float: Silhouette score
    """
    from sklearn.metrics import silhouette_score
    return silhouette_score(X, cluster_labels)
