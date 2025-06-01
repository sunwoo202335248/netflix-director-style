# preprocessing.py
def preprocess_netflix_data(csv_path):
    """
    넷플릭스 원본 CSV 파일을 전처리하여 분석 가능한 데이터프레임을 생성합니다.

    Args:
        csv_path (str): 원본 CSV 파일 경로 (예: netflix_titles.csv)

    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    df = pd.read_csv(csv_path)

    # 사용할 열 선택
    columns = [
        'show_id', 'type', 'title', 'director', 'cast', 'country',
        'date_added', 'release_year', 'rating', 'duration', 'listed_in', 'description'
    ]
    df = df[columns]

    # 1. 주요 열 기준 결측치 제거
    df = df.dropna(subset=['director', 'rating', 'duration', 'listed_in'])

    # 2. duration을 숫자로 변환
    def extract_duration(value):
        try:
            return int(value.strip().split(' ')[0])
        except:
            return None
    df['duration_number'] = df['duration'].apply(extract_duration)
    df = df.dropna(subset=['duration_number'])

    # 3. 장르 통합 함수
    def map_genre(genre):
        genre = genre.lower()
        if 'drama' in genre or 'romantic' in genre:
            return 'Drama'
        elif 'comedy' in genre:
            return 'Comedy'
        elif 'documentary' in genre:
            return 'Documentary'
        elif 'action' in genre or 'thriller' in genre or 'horror' in genre:
            return 'Action'
        elif 'children' in genre or 'kids' in genre or 'family' in genre:
            return 'Children'
        else:
            return 'Other'

    # 4. listed_in의 첫 번째 장르 기준으로 대표 장르 분류
    df['primary_genre_grouped'] = df['listed_in'].apply(lambda x: map_genre(x.split(',')[0]))

    # 5. 출연진 수 파생변수 추가
    df['cast_count'] = df['cast'].apply(lambda x: len(x.split(',')) if pd.notnull(x) else 0)

    # 6. rating 인코딩
    rating_encoder = LabelEncoder()
    df['rating_encoded'] = rating_encoder.fit_transform(df['rating'])

    # 7. One-Hot Encoding (장르 및 타입)
    df = pd.get_dummies(df, columns=['primary_genre_grouped', 'type'], prefix=['genre', 'type'])

    # 8. duration 정규화
    scaler = StandardScaler()
    df['duration_scaled'] = scaler.fit_transform(df[['duration_number']])

    return df



# Clustering_KMeans.py
def cluster_directors(df, n_clusters=4):
    """
    감독별 평균 콘텐츠 특성을 바탕으로 KMeans 클러스터링을 수행합니다.

    Args:
        df (pd.DataFrame): 전처리된 Netflix 데이터프레임
        n_clusters (int): 클러스터 수

    Returns:
        pd.DataFrame: 클러스터 라벨이 추가된 감독 통계 테이블
    """
    from sklearn.cluster import KMeans
    # 클러스터링 로직 생략
    return df


# Classification_DecisionTree.py
def train_decision_tree(X, y, max_depth=5):
    """
    콘텐츠 특성을 기반으로 장르를 분류하는 결정 트리 모델을 학습합니다.

    Args:
        X (pd.DataFrame): 입력 피처
        y (pd.Series): 장르 라벨
        max_depth (int): 최대 트리 깊이

    Returns:
        DecisionTreeClassifier: 학습된 모델
    """
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X, y)
    return model


# Random_Forest.py
def train_random_forest(X_train, y_train):
    """
    콘텐츠 기반 랜덤 포레스트 분류 모델을 학습합니다.

    Args:
        X_train (pd.DataFrame): 학습용 입력 피처
        y_train (pd.Series): 학습용 라벨

    Returns:
        RandomForestClassifier: 학습된 모델
    """
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_split=5,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf


# GridSearch_Tuning.py
def tune_hyperparameters(X_train, y_train, param_grid):
    """
    랜덤 포레스트 모델의 하이퍼파라미터를 GridSearchCV로 탐색합니다.

    Args:
        X_train (pd.DataFrame): 학습용 입력 피처
        y_train (pd.Series): 학습용 라벨
        param_grid (dict): 탐색할 하이퍼파라미터 설정

    Returns:
        GridSearchCV: 최적 파라미터를 포함한 GridSearch 객체
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(RandomForestClassifier(random_state=42),
                        param_grid=param_grid, cv=5, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid


# cross_validation.py
def run_cross_validation(X, y, cv=10):
    """
    교차검증을 통해 모델의 일반화 성능(F1-macro)을 평가합니다.

    Args:
        X (pd.DataFrame): 입력 피처
        y (pd.Series): 라벨
        cv (int): 폴드 수

    Returns:
        list: 각 폴드별 F1 점수 리스트
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import make_scorer, f1_score
    from sklearn.model_selection import cross_val_score
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    f1_macro = make_scorer(f1_score, average='macro')
    scores = cross_val_score(model, X, y, cv=cv, scoring=f1_macro)
    return scores


# Evaluation_Metrics.py
def evaluate_classification(y_true, y_pred, target_names=None):
    """
    분류 모델의 정확도, 혼동 행렬, 리포트를 출력하고 시각화합니다.
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import seaborn as sns
    import matplotlib.pyplot as plt

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")
    print("Confusion Matrix: ", cm)
    print("Classification Report: ", classification_report(y_true, y_pred, target_names=target_names))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def evaluate_clustering(X, cluster_labels):
    """
    클러스터링 결과의 실루엣 점수를 계산합니다.
    """
    from sklearn.metrics import silhouette_score
    return silhouette_score(X, cluster_labels)
