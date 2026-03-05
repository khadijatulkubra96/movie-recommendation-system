import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title('🎬 Movie Recommender System')
st.markdown("Find movies similar to your favorites using our ML model!")

movies = pickle.load(open('movie_list.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

# Recommendation Function 
def recommend(movie_title):
    idx = movies[movies['title_clean'] == movie_title.lower().strip()].index[0]
    
    # Similarity scores
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    
    recommended_movies = []
    for i in distances[1:6]: # Top 5 movies
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

# (User Interface)
selected_movie = st.selectbox(
    "Select a movie you liked:",
    movies['title'].values
)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    st.subheader("You might also like:")
    for i in recommendations:
        st.success(f"🎥 {i}")