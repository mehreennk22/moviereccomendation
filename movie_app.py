import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
@st.cache_data
def load_data():
    movies_df = pd.read_csv("movies.csv")
    ratings_df = pd.read_csv("ratings.csv")
    return movies_df, ratings_df

movies_df, ratings_df = load_data()

st.title("🎬 Movie Recommendation System")

# Show the dataset (optional)
if st.checkbox("Show sample movie data"):
    st.dataframe(movies_df.head())

# Content-based filtering using genres
st.subheader("🔍 Get Recommendations Based on Movie Genres")

# Handle missing or invalid data
if 'genres' not in movies_df.columns or movies_df['genres'].isnull().all():
    st.error("No genre data available in movies.csv")
else:
    tfidf = TfidfVectorizer(stop_words='english')
    movies_df['genres'] = movies_df['genres'].fillna('')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

    def recommend(title, num_recommendations=5):
        if title not in indices:
            return ["Movie not found in database."]
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
        movie_indices = [i[0] for i in sim_scores]
        return movies_df['title'].iloc[movie_indices]

    # User input
    movie_input = st.text_input("Enter a movie title to get recommendations:")

    if movie_input:
        recommendations = recommend(movie_input)
        st.subheader("Recommended Movies:")
        for rec in recommendations:
            st.write(f"• {rec}")
