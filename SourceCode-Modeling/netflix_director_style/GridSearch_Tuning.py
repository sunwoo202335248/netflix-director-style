import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 데이터 불러오기
df = pd.read_csv("netflix_preprocessed_final.csv")

# 장르 컬럼 추출 (One-Hot에서 복원)
# genre_Drama, genre_Comedy, ... → 원래 라벨로 복원
genre_columns = [col for col in df.columns if col.startswith("genre_")]
df["primary_genre_label"] = df[genre_columns].idxmax(axis=1).str.replace("genre_", "")

# 라벨 인코딩
from sklearn.preprocessing import LabelEncoder
genre_encoder = LabelEncoder()
df["genre_encoded"] = genre_encoder.fit_transform(df["primary_genre_label"])

# 입력(X), 출력(y) 설정
X = df[["duration_scaled", "rating_encoded", "cast_count"]]  # 필요시 더 추가
y = df["genre_encoded"]

# 학습/검증 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 파라미터 그리드 정의
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "class_weight": ["balanced"]
}

# GridSearchCV 실행
clf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(clf, param_grid, cv=5, scoring="f1_macro", n_jobs=-1)
grid.fit(X_train, y_train)

# 결과 출력
print("✅ Best Parameters:", grid.best_params_)
print("✅ Best Cross-Validation f1_score: {:.2f}%".format(grid.best_score_ * 100))

# 최적 모델로 테스트셋 평가
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=genre_encoder.classes_))
