import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'title': [
        'Titanic', 'Avatar', 'The Notebook',
        'Inception', 'Interstellar', 'The Dark Knight'
    ],
    'genre': [
        'romance drama', 'sci-fi adventure', 'romance drama',
        'sci-fi thriller', 'sci-fi space', 'action crime'
    ]
}

df = pd.DataFrame(data)

cv = CountVectorizer()
matrix = cv.fit_transform(df['genre'])

similarity = cosine_similarity(matrix)

def recommend_movie(movie):
    if movie not in df['title'].values:
        print("Movie not found in database.")
        return

    index = df[df['title'] == movie].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("\nRecommended Movies:")
    for i in scores[1:4]:
        print(df.iloc[i[0]].title)