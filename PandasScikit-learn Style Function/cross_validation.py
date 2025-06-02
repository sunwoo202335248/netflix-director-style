def run_cross_validation(X, y, cv=10):
    """
    Evaluate a Decision Tree classifier using K-fold cross-validation.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray
        Target labels.
    cv : int, optional (default=10)
        Number of folds in cross-validation.

    Returns
    -------
    list of float
        F1-macro scores for each fold.
    """
