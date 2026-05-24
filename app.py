import streamlit as st
import pickle
import pandas as pd

# =========================
# LOAD MODELS
# =========================
user_item = pickle.load(open("user_item.pkl", "rb"))
user_similarity = pickle.load(open("user_similarity.pkl", "rb"))
movie_user = pickle.load(open("movie_user.pkl", "rb"))
knn_model = pickle.load(open("knn.pkl", "rb"))

data = pickle.load(open("data.pkl", "rb"))          # 🔥 correct
movies = pickle.load(open("movies.pkl", "rb"))
content_similarity = pickle.load(open("content.pkl", "rb"))
indices = pickle.load(open("indices.pkl", "rb"))

# =========================
# USER SIMILARITY DF
# =========================
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item.index,
    columns=user_item.index
)

# =========================
# WEIGHTED CF
# =========================
def recommend_cf(user_id, n=5):

    if user_id not in user_item.index:
        return data.groupby('title')['rating'].mean().sort_values(ascending=False).head(n).index.tolist()

    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:11]
    similar_ratings = user_item.loc[similar_users.index]
    sim_scores = user_similarity_df[user_id].loc[similar_users.index]

    weighted_scores = (similar_ratings.T * sim_scores).T.sum() / sim_scores.sum()
    weighted_scores = weighted_scores.sort_values(ascending=False)

    watched = user_item.loc[user_id]
    weighted_scores = weighted_scores[watched == 0]

    return weighted_scores.head(n).index.tolist()

# =========================
# CONTENT
# =========================
def recommend_content(title, n=5):

    if title not in indices:
        return ["Movie not found"]

    idx = indices[title]

    sim_scores = list(enumerate(content_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]

    movie_indices = [i[0] for i in sim_scores]

    return movies['title'].iloc[movie_indices].tolist()

# =========================
# HYBRID
# =========================
def hybrid_recommend(user_id, title, n=5):
    return list(dict.fromkeys(
        recommend_cf(user_id, 10) + recommend_content(title, 10)
    ))[:n]

# =========================
# KNN
# =========================
def recommend_knn(movie_name, n=5):

    if movie_name not in movie_user.index:
        return ["Movie not found"]

    idx = movie_user.index.get_loc(movie_name)

    distances, indices_knn = knn_model.kneighbors(
        movie_user.values[idx].reshape(1, -1),
        n_neighbors=n+1
    )

    return [movie_user.index[i] for i in indices_knn.flatten()[1:]]

# =========================
# UI
# =========================
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

st.title("🎬 Movie Recommendation System")
st.markdown("### Discover movies you’ll love ❤️")

option = st.sidebar.selectbox(
    "Choose Model",
    ["Collaborative Filtering", "Content-Based", "Hybrid", "KNN"]
)

# =========================
# UI LOGIC
# =========================
if option == "Collaborative Filtering":
    user_id = st.number_input("Enter User ID", min_value=1)

    if st.button("Recommend"):
        recs = recommend_cf(user_id)
        for i, m in enumerate(recs, 1):
            st.markdown(f"**{i}. {m}**")

elif option == "Content-Based":
    movie = st.text_input("Enter Movie Name")

    if st.button("Recommend"):
        recs = recommend_content(movie)
        for i, m in enumerate(recs, 1):
            st.markdown(f"**{i}. {m}**")

elif option == "Hybrid":
    user_id = st.number_input("Enter User ID", min_value=1)
    movie = st.text_input("Enter Movie Name")

    if st.button("Recommend"):
        recs = hybrid_recommend(user_id, movie)
        for i, m in enumerate(recs, 1):
            st.markdown(f"**{i}. {m}**")

elif option == "KNN":
    movie = st.text_input("Enter Movie Name")

    if st.button("Recommend"):
        recs = recommend_knn(movie)
        for i, m in enumerate(recs, 1):
            st.markdown(f"**{i}. {m}**")