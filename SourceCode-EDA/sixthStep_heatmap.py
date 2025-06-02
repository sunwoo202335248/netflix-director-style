import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns



# Load the preprocessed Netflix dataset
df = pd.read_csv("netflix_preprocessed_final.csv")


# Reconstruct primary genre labels from one-hot encoded genre columns
genre_cols = [col for col in df.columns if col.startswith("genre_")]
df["primary_genre_grouped"] = df[genre_cols].idxmax(axis=1).str.replace("genre_", "")


# Apply label encoding to convert categorical genres to numerical formatt
genre_encoder = LabelEncoder()
df["genre_encoded"] = genre_encoder.fit_transform(df["primary_genre_grouped"])

# Build Random Forest classification model to predict genres
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


X = df[["duration_scaled", "rating_encoded", "cast_count"]]
y = df["genre_encoded"]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
clf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X)


# Convert encoded predictions back to original genre labels
# inverse_transform() converts numerical predictions to readable genre names
df["predicted_genre"] = genre_encoder.inverse_transform(y_pred)


#analyze prediced genre distribution by director
#calculate normalizaed value counts to show proportions 
genre_dist = df.groupby("director")["predicted_genre"].value_counts(normalize=True).unstack().fillna(0)


# Convert encoded predictions back to original genre labels
# inverse_transform() converts numerical predictions to readable genre names
top_directors = df["director"].value_counts().head(10).index
top_data = genre_dist.loc[top_directors]


# visualize heatmap
plt.figure(figsize=(12, 6))

#
sns.heatmap(top_data, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Top Directors' Predicted Genre Distribution")
plt.xlabel("Predicted Genre")
plt.ylabel("Director")
plt.tight_layout()
plt.show()



