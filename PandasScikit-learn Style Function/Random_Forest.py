def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier on the given training data.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Feature matrix for training.
    y_train : pd.Series or np.ndarray
        Training labels.

    Returns
    -------
    RandomForestClassifier
        A fitted RandomForestClassifier instance with predefined parameters.

    Notes
    -----
    The model uses:
    - 100 estimators
    - max_depth=7
    - class_weight='balanced'
    """
