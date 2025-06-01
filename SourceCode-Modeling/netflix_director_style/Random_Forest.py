import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


# 1. 데이터 불러오기
df = pd.read_csv("netflix_preprocessed_final.csv")

# 2. 장르 복원 (One-Hot → 문자열)
genre_cols = [col for col in df.columns if col.startswith("genre_")]
df["primary_genre_grouped"] = df[genre_cols].idxmax(axis=1).str.replace("genre_", "")

# 3. 장르 인코딩
genre_encoder = LabelEncoder()
df["genre_encoded"] = genre_encoder.fit_transform(df["primary_genre_grouped"])

# 4. 입력 데이터 선택 (콘텐츠 단위)
X = df[["duration_scaled", "rating_encoded", "cast_count"]]
y = df["genre_encoded"]

# 5. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 모델 학습 (콘텐츠 단위 + 가중치 적용)
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=7,
    min_samples_split=5,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train, y_train)

# 7. 예측 및 평가
y_pred = clf.predict(X_test)
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=genre_encoder.classes_))


# Confusion Matrix 숫자 출력
cm = confusion_matrix(y_test, y_pred)
print("\n📊 Confusion Matrix:")
print(cm)
