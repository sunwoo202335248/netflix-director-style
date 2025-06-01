import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

#  1. Load the preprocessed Netflix dataset
df = pd.read_csv("netflix_preprocessed_final.csv")

#  2. Reconstruct original genre labels from one-hot encoded columns
# Example: genre_Drama, genre_Comedy → Drama, Comedy
genre_columns = [col for col in df.columns if col.startswith("genre_")]
df["primary_genre_label"] = df[genre_columns].idxmax(axis=1).str.replace("genre_", "")

#  3. Encode genre labels into numeric format (e.g., 'Action' → 0, 'Drama' → 1, ...)
genre_encoder = LabelEncoder()
df["genre_encoded"] = genre_encoder.fit_transform(df["primary_genre_label"])

#  4. Select input features and target label
# Features: normalized duration, encoded rating, cast size
X = df[["duration_scaled", "rating_encoded", "cast_count"]]
y = df["genre_encoded"]

#  5. Split the data into training and test sets (80:20 ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  6. Define the hyperparameter search space for RandomForest
param_grid = {
    "n_estimators": [100, 200],       # Number of trees in the forest
    "max_depth": [5, 10, None],       # Maximum tree depth (None = unlimited)
    "min_samples_split": [2, 5],      # Minimum samples required to split an internal node
    "min_samples_leaf": [1, 2],       # Minimum samples required at a leaf node
    "class_weight": ["balanced"]      # Adjusts weights inversely proportional to class frequencies
}

#  7. Run GridSearchCV with 5-fold cross-validation and macro F1 scoring
clf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(
    clf,
    param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1  # Use all available CPU cores
)
grid.fit(X_train, y_train)

#  8. Output the best parameter combination and cross-validation score
print(" Best Parameters:", grid.best_params_)
print(" Best Cross-Validation f1_score: {:.2f}%".format(grid.best_score_ * 100))

#  9. Evaluate the best model on the test set
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

#  10. Print detailed classification results by genre
print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=genre_encoder.classes_))
