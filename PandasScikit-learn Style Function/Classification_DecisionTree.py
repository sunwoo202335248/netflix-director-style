def train_decision_tree(X, y, max_depth=5):
    """
    Train a Decision Tree classifier on the given dataset.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray
        Target labels.
    max_depth : int, optional (default=5)
        Maximum depth of the decision tree.

    Returns
    -------
    DecisionTreeClassifier
        A trained DecisionTreeClassifier instance.
    """
