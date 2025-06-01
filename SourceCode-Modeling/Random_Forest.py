import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

#  1. Load the preprocessed Netflix dataset
df = pd.read_csv("netflix_preprocessed_final.csv")

#  2. Reconstruct original genre labels from one-hot encoded columns
genre_cols = [col for col in df.columns if col.startswith("genre_")]
df["primary_genre_grouped"] = df[genre_cols].idxmax(axis=1).str.replace("genre_", "")

#  3. Encode genre labels into numeric values
genre_encoder = LabelEncoder()
df["genre_encoded"] = genre_encoder.fit_transform(df["primary_genre_grouped"])

#  4. Define input features (duration, rating, cast size) and target variable
X = df[["duration_scaled", "rating_encoded", "cast_count"]]
y = df["genre_encoded"]

#  5. Split the dataset into training and test sets (80:20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  6. Initialize and train a Random Forest Classifier with balanced class weights
clf = RandomForestClassifier(
    n_estimators=100,         # Number of trees
    max_depth=7,              # Maximum depth of each tree
    min_samples_split=5,      # Minimum number of samples required to split an internal node
    min_samples_leaf=1,       # Minimum number of samples required at a leaf node
    class_weight='balanced',  # Handle class imbalance automatically
    random_state=42
)
clf.fit(X_train, y_train)

#  7. Predict the genre for test data and evaluate performance
y_pred = clf.predict(X_test)

# Print classification performance metrics
print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=genre_encoder.classes_))

# Print confusion matrix (raw counts)
cm = confusion_matrix(y_test, y_pred)
print("\n Confusion Matrix:")
print(cm)
