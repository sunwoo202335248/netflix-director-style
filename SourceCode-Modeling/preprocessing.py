import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv("netflix_titles.csv")

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

# 4. listed_in의 첫 번째 장르를 기준으로 그룹화(여러 장르들을 비슷한 장르끼리 나눠서 5-6개의 대표 그룹으로 바꾸는 것)
df['primary_genre_grouped'] = df['listed_in'].apply(lambda x: map_genre(x.split(',')[0]))

# 5. 출연진 수 파생변수 추가(출연한 사람 수를 cast_count에 저장함)
df['cast_count'] = df['cast'].apply(lambda x: len(x.split(',')) if pd.notnull(x) else 0)

# 6. rating 인코딩 (등급을 숫자로 변환하여 rating_encoded에 저장함)
rating_encoder = LabelEncoder()
df['rating_encoded'] = rating_encoder.fit_transform(df['rating'])

# 7. One-Hot Encoding (primary_genre_grouped, type)(장르와 콘텐츠 타입을 0/1로 변환함)
df = pd.get_dummies(df, columns=['primary_genre_grouped', 'type'], prefix=['genre', 'type'])

# 8. duration 정규화
scaler = StandardScaler()
df['duration_scaled'] = scaler.fit_transform(df[['duration_number']])

df.to_csv("netflix_preprocessed_final.csv", index=False)
print("전처리 완료: netflix_preprocessed_final.csv")
