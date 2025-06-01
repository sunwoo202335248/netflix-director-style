from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 🔹 1. Load preprocessed data and prepare labels
df = pd.read_csv("netflix_preprocessed_final.csv")

# Extract genre column names (e.g., genre_Drama, genre_Action, ...)
genre_cols = [col for col in df.columns if col.startswith("genre_")]

# Reconstruct genre labels by finding the column with highest value (i.e., the one-hot encoded genre)
df["primary_genre_grouped"] = df[genre_cols].idxmax(axis=1).str.replace("genre_", "")

# Encode genre labels to numeric values
genre_encoder = LabelEncoder()
df["genre_encoded"] = genre_encoder.fit_transform(df["primary_genre_grouped"])

# Select features and target
X = df[["duration_scaled", "rating_encoded", "cast_count"]]
y = df["genre_encoded"]

# 2. Define Decision Tree model (with limited depth)
clf = DecisionTreeClassifier(max_depth=5, random_state=42)

# 3. Define evaluation metric: F1-score with macro averaging
f1_macro = make_scorer(f1_score, average='macro')

# Perform 10-fold cross-validation using F1-macro as scoring
scores = cross_val_score(clf, X, y, cv=10, scoring=f1_macro)

# 4. Display results
print(" F1-macro score for each fold:", np.round(scores, 3))
print(" Average F1-macro score: {:.2f}%".format(np.mean(scores) * 100))
