import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the original Netflix dataset
df = pd.read_csv("netflix_titles.csv")

# Select relevant columns only
columns = [
    'show_id', 'type', 'title', 'director', 'cast', 'country',
    'date_added', 'release_year', 'rating', 'duration', 'listed_in', 'description'
]
df = df[columns]

# 1. Remove rows with missing values in key columns
# These fields are essential for modeling and feature engineering
df = df.dropna(subset=['director', 'rating', 'duration', 'listed_in'])

# 2. Convert 'duration' (e.g., "90 min", "1 Season") to numeric format
# Keep only numeric part (e.g., 90 from "90 min")
def extract_duration(value):
    try:
        return int(value.strip().split(' ')[0])
    except:
        return None

df['duration_number'] = df['duration'].apply(extract_duration)

# Remove rows where 'duration_number' could not be extracted
df = df.dropna(subset=['duration_number'])

# 3. Map listed genres to a smaller number of representative categories
# This simplifies the variety of genre combinations to a manageable set
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

# 4. Extract the first genre from 'listed_in' and map it to grouped category
df['primary_genre_grouped'] = df['listed_in'].apply(lambda x: map_genre(x.split(',')[0]))

# 5. Create a new feature: number of cast members in the content
df['cast_count'] = df['cast'].apply(lambda x: len(x.split(',')) if pd.notnull(x) else 0)

# 6. Encode 'rating' labels as numeric values (e.g., "TV-MA" → 7)
rating_encoder = LabelEncoder()
df['rating_encoded'] = rating_encoder.fit_transform(df['rating'])

# 7. Apply One-Hot Encoding to 'primary_genre_grouped' and 'type' (e.g., genre_Drama, type_Movie)
# This converts categorical variables into binary columns for modeling
df = pd.get_dummies(df, columns=['primary_genre_grouped', 'type'], prefix=['genre', 'type'])

# 8. Normalize the 'duration_number' using standard scaling (mean=0, std=1)
scaler = StandardScaler()
df['duration_scaled'] = scaler.fit_transform(df[['duration_number']])

# Save the final preprocessed dataset to CSV
df.to_csv("netflix_preprocessed_final.csv", index=False)
print("Preprocessing complete: netflix_preprocessed_final.csv")
