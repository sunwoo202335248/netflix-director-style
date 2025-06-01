from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 🔹 1. 데이터 로드 및 전처리
df = pd.read_csv("netflix_preprocessed_final.csv")
genre_cols = [col for col in df.columns if col.startswith("genre_")]
df["primary_genre_grouped"] = df[genre_cols].idxmax(axis=1).str.replace("genre_", "")

genre_encoder = LabelEncoder()
df["genre_encoded"] = genre_encoder.fit_transform(df["primary_genre_grouped"])

X = df[["duration_scaled", "rating_encoded", "cast_count"]]
y = df["genre_encoded"]

# ✅ 2. 모델 정의
clf = DecisionTreeClassifier(max_depth=5, random_state=42)

# ✅ 3. F1-macro 기준 교차검증 수행
f1_macro = make_scorer(f1_score, average='macro')
scores = cross_val_score(clf, X, y, cv=10, scoring=f1_macro)

# ✅ 4. 결과 출력
print("📊 각 Fold F1-macro:", np.round(scores, 3))
print("✅ 평균 F1-macro: {:.2f}%".format(np.mean(scores) * 100))
