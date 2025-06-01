import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    silhouette_score
)

def evaluate_classification(y_true, y_pred, target_names=None):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=target_names)

    print("✅ Accuracy: {:.2f}%".format(acc * 100))
    print("\n✅ Confusion Matrix:\n", cm)
    print("\n✅ Classification Report:\n", report)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

def evaluate_clustering(X, cluster_labels):
    score = silhouette_score(X, cluster_labels)
    print("✅ Silhouette Score: {:.3f}".format(score))
    return score

# 📌 테스트 실행 (넷플릭스 분류 모델 평가용)
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    # 1. 데이터 로드
    df = pd.read_csv("netflix_preprocessed_final.csv")

    # 2. 장르 복원 (One-Hot → 문자열)
    genre_cols = [col for col in df.columns if col.startswith("genre_")]
    df["primary_genre_grouped"] = df[genre_cols].idxmax(axis=1).str.replace("genre_", "")

    # 3. 장르 인코딩
    genre_encoder = LabelEncoder()
    df["genre_encoded"] = genre_encoder.fit_transform(df["primary_genre_grouped"])

    # 4. 입력 피처 및 라벨
    X = df[["duration_scaled", "rating_encoded", "cast_count"]]
    y = df["genre_encoded"]

    # 5. 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # 6. Random Forest 모델 학습
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 7. 평가 실행
    evaluate_classification(y_test, y_pred, target_names=genre_encoder.classes_)
