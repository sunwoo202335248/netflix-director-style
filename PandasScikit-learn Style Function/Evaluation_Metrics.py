def evaluate_classification(y_true, y_pred, target_names=None):
    """
    Evaluate classification results and display metrics and confusion matrix.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    target_names : list of str, optional
        Names of target classes to display in the classification report.

    Returns
    -------
    None
    """

def evaluate_clustering(X, cluster_labels):
    """
    Compute the Silhouette Score of a clustering result.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The feature set used for clustering.
    cluster_labels : array-like of shape (n_samples,)
        Cluster labels assigned to each sample.

    Returns
    -------
    float
        Silhouette score of the clustering result.
    """
