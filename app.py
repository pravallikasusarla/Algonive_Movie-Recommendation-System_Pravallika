# Your Streamlit app code starts here

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load data
movies = pd.read_csv("moviedata/movies.csv")
ratings = pd.read_csv("moviedata/ratings.csv")

movies['genres_clean'] = movies['genres'].fillna('').apply(lambda s: s.replace('|', ' '))
movies['content'] = movies['title'].fillna('') + ' ' + movies['genres_clean']

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(movies['content'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def recommend_movie(title, topn=5):
    if title not in indices:
        return None
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:topn+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['title','genres']]

st.title("🎬 Movie Recommendation System")
st.write("Get movie recommendations based on your favorite film!")

movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie:", movie_list)

if st.button("Recommend"):
    results = recommend_movie(selected_movie, topn=10)
    if results is not None:
        st.subheader("Top Recommendations:")
        for _, row in results.iterrows():
            st.write(f"🎥 **{row.title}** — *{row.genres}*")
    else:
        st.warning("Movie not found. Please try another one.")
# ============================
# 🎬 MOVIE RECOMMENDATION SYSTEM
# Content-Based + User-Based
# ============================

#Import Libraries 
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import numpy as np

#Load Dataset
movies = pd.read_csv("moviedata/movies.csv")
ratings = pd.read_csv("moviedata/ratings.csv")

# Clean and prepare movie content
movies['genres_clean'] = movies['genres'].fillna('').apply(lambda s: s.replace('|', ' '))
movies['content'] = movies['title'].fillna('') + ' ' + movies['genres_clean']

#CONTENT-BASED RECOMMENDER
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(movies['content'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def recommend_movie(title, topn=5):
    """Recommend movies similar to a given title"""
    if title not in indices:
        return None
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:topn+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['title', 'genres']]

#USER-BASED RECOMMENDER
# Create user-movie matrix
user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

def recommend_user_based(user_id, topn=5):
    """Recommend movies based on similar users"""
    if user_id not in user_similarity_df.index:
        return f"User {user_id} not found."

    # Find top similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]
    similar_users_ids = similar_users.index

    # Movies rated by similar users
    similar_users_ratings = ratings[ratings['userId'].isin(similar_users_ids)]
    user_movies = ratings[ratings['userId'] == user_id]['movieId'].unique()

    # Weighted score calculation
    weighted_scores = similar_users_ratings.merge(
        pd.DataFrame(similar_users),
        left_on='userId', right_index=True
    )
    weighted_scores['weighted_rating'] = weighted_scores['rating'] * weighted_scores[user_id]

    recommendation_scores = weighted_scores.groupby('movieId')['weighted_rating'].sum() / \
                             weighted_scores.groupby('movieId')[user_id].sum()
    recommendation_scores = recommendation_scores.dropna().sort_values(ascending=False)

    # Recommend unseen movies
    recommended_movie_ids = [mid for mid in recommendation_scores.index if mid not in user_movies][:topn]
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)][['movieId', 'title', 'genres']]
    return recommended_movies

#STREAMLIT UI
st.title("🎬 Movie Recommendation System")

# Tabs for different recommendation types
tab1, tab2 = st.tabs(["🎥 Content-Based", "👥 User-Based"])

# ---- TAB 1: Content-Based ----
with tab1:
    st.header("🎥 Content-Based Recommendations")
    movie_list = movies['title'].values
    # 👇 Add a unique key here
    selected_movie = st.selectbox("Select a movie:", movie_list, key="content_movie_select")

    if st.button("Recommend Similar Movies", key="content_button"):
        results = recommend_movie(selected_movie, topn=10)
        if results is not None:
            st.dataframe(results)
        else:
            st.warning("Movie not found. Try another title.")

# ---- TAB 2: User-Based ----
with tab2:
    st.header("👥 User-Based Recommendations")
    # 👇 Add unique keys here too
    user_id = st.number_input("Enter User ID:", 
                              min_value=1, 
                              max_value=int(ratings['userId'].max()), 
                              key="user_input")

    if st.button("Get User-Based Recommendations", key="user_button"):
        recs = recommend_user_based(user_id, topn=10)
        if isinstance(recs, pd.DataFrame):
            st.dataframe(recs)
        else:
            st.warning(recs)