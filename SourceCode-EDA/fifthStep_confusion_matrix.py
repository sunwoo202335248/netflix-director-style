import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score


#load the data 
df = pd.read_csv("netflix_preprocessed_final.csv")
genre_cols = [col for col in df.columns if col.startswith("genre_")]
df["primary_genre_grouped"] = df[genre_cols].idxmax(axis=1).str.replace("genre_", "")
genre_encoder = LabelEncoder()
df["genre_encoded"] = genre_encoder.fit_transform(df["primary_genre_grouped"])
X = df[["duration_scaled", "rating_encoded", "cast_count"]]
y = df["genre_encoded"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# do random forest models
rf = RandomForestClassifier(random_state=42, class_weight="balanced", min_samples_split=5)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)


# create two confusion matrix heatmap for comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#visulize confusion matrix for random forest model
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_rf, display_labels=genre_encoder.classes_, # true label and random forest label predicton, Use original genre names as axis labels
    cmap="Blues", ax=axes[0], xticks_rotation=45# Blue color scheme for visual distinction, Plot on left subplot Rotate x-axis labels 45° for readability
)
axes[0].set_title("Random Forest")

# create confusion matrix for decision tree
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_dt, display_labels=genre_encoder.classes_,
    cmap="Oranges", ax=axes[1], xticks_rotation=45
)
axes[1].set_title("Decision Tree")

#automatically adjust two subplots 
plt.tight_layout()
plt.show()


# output accuracy for tow models
print("Random Forest Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred_rf) * 100))
print("Decision Tree Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred_dt) * 100))



