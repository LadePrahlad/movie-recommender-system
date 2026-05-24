import numpy as np
import pandas as pd
import pickle
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# LOAD DATA
# =========================
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# =========================
# DATA CLEANING
# =========================
movie_counts = ratings['movieId'].value_counts()
popular_movies_ids = movie_counts[movie_counts > 50].index
ratings = ratings[ratings['movieId'].isin(popular_movies_ids)]

user_counts = ratings['userId'].value_counts()
active_users = user_counts[user_counts > 50].index
ratings = ratings[ratings['userId'].isin(active_users)]

# =========================
# MERGE DATA
# =========================
data = ratings.merge(movies, on='movieId')

# =========================
# USER-ITEM MATRIX
# =========================
user_item = data.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# =========================
# USER SIMILARITY
# =========================
user_sparse = csr_matrix(user_item.values)
user_similarity = cosine_similarity(user_sparse)

# =========================
# CONTENT-BASED
# =========================
movies = movies.reset_index(drop=True)
movies['genres'] = movies['genres'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

content_similarity = cosine_similarity(tfidf_matrix)

indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# =========================
# KNN MODEL
# =========================
movie_user = data.pivot_table(index='title', columns='userId', values='rating').fillna(0)
movie_sparse = csr_matrix(movie_user.values)

knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(movie_sparse)

# =========================
# SAVE EVERYTHING
# =========================
pickle.dump(user_item, open("user_item.pkl", "wb"))
pickle.dump(user_similarity, open("user_similarity.pkl", "wb"))
pickle.dump(movie_user, open("movie_user.pkl", "wb"))
pickle.dump(knn_model, open("knn.pkl", "wb"))

pickle.dump(data, open("data.pkl", "wb"))   # 🔥 important fix
pickle.dump(movies, open("movies.pkl", "wb"))
pickle.dump(content_similarity, open("content.pkl", "wb"))
pickle.dump(indices, open("indices.pkl", "wb"))

print("✅ Models saved successfully!")