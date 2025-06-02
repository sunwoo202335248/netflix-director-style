def preprocess_netflix_data(csv_path):
    """
    Load and preprocess the Netflix dataset for modeling.

    Parameters
    ----------
    csv_path : str
        Path to the original Netflix CSV file (e.g., "netflix_titles.csv").

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame containing encoded and scaled features,
        suitable for modeling and analysis.

    Notes
    -----
    The function performs the following steps:
    - Removes rows with missing values in key columns.
    - Extracts numeric duration.
    - Maps multiple genres to a reduced number of genre groups.
    - Computes the number of cast members.
    - Label-encodes ratings.
    - Applies One-Hot Encoding to genre and content type.
    - Standardizes the duration feature.
    """
