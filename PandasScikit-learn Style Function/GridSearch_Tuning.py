def tune_hyperparameters(X_train, y_train, param_grid):
    """
    Perform hyperparameter tuning using GridSearchCV on a Random Forest model.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Target labels.
    param_grid : dict
        Dictionary of parameter ranges to search.

    Returns
    -------
    GridSearchCV
        The fitted GridSearchCV instance containing the best estimator and scores.
    """
