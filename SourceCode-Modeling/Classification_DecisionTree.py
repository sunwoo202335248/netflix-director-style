import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load preprocessed dataset
df = pd.read_csv("netflix_preprocessed_final.csv")

# 2. Restore genre labels from one-hot encoded columns (e.g., genre_Drama → Drama)
genre_cols = [col for col in df.columns if col.startswith("genre_")]
df["primary_genre_grouped"] = df[genre_cols].idxmax(axis=1).str.replace("genre_", "")

# 3. Encode genre labels as integers
genre_encoder = LabelEncoder()
df["genre_encoded"] = genre_encoder.fit_transform(df["primary_genre_grouped"])

# 4. Select input features and target variable
X = df[["duration_scaled", "rating_encoded", "cast_count"]]
y = df["genre_encoded"]

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train a Decision Tree Classifier
clf = DecisionTreeClassifier(
    max_depth=7,
    min_samples_split=5,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train, y_train)

# 7. Predict and evaluate
y_pred = clf.predict(X_test)

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=genre_encoder.classes_))

# Optional: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n📊 Confusion Matrix:")
print(cm)
