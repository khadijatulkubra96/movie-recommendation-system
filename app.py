import streamlit as st
import pickle
import pandas as pd
import requests

# Function to fetch poster from API
def fetch_poster(movie_id):
    # Standard TMDb API URL
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    try:
        data = requests.get(url).json()
        poster_path = data['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        return full_path
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster+Available"

# Page Config 
st.set_page_config(page_title="Movie Lounge", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stHeading h1 { color: #ff4b4b; font-family: 'Helvetica Neue', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

st.title('🎬 Movie Lounge: AI-Powered Recommendations')

# Load Data
movies = pickle.load(open('movie_list.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

selected_movie = st.selectbox(
    "Search for a movie you've watched...",
    movies['title'].values
)

if st.button('Get Recommendations'):
    # Recommendation Logic
    idx = movies[movies['title_clean'] == selected_movie.lower().strip()].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    
    st.subheader(f"Because you liked '{selected_movie}', we suggest:")
    
    # Creating 5 Columns for Posters
    cols = st.columns(5)
    
    for i in range(1, 6):
        movie_idx = distances[i][0]
        tmdb_id = movies.iloc[movie_idx].id 
        title = movies.iloc[movie_idx].title
        rating = movies.iloc[movie_idx].vote_average
        
        with cols[i-1]:
            poster = fetch_poster(tmdb_id)
            st.image(poster)
            st.markdown(f"**{title}**")
            st.caption(f"⭐ Rating: {rating}/10")
