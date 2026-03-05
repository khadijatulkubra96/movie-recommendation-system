import streamlit as st
import pickle
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Helper Function: Fetch Poster ---
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    try:
        data = requests.get(url).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        return "https://via.placeholder.com/500x750?text=No+Poster"
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster"

# --- Page Layout & Styling ---
st.set_page_config(page_title="Movie Lounge", layout="wide")

# CSS for Background Image and Centered Text
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                    url('https://images.unsplash.com/photo-1536440136628-849c177e76a1?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
        background-attachment: fixed;
        color: white;
    }
    .main-title {
        font-size: 60px;
        font-weight: bold;
        text-align: center;
        color: #E50914; /* Netflix Red */
        margin-top: -50px;
    }
    .sub-title {
        font-size: 20px;
        text-align: center;
        margin-bottom: 50px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_data():
    movies = pickle.load(open('movie_list.pkl', 'rb'))
    if 'tags' not in movies.columns:
        movies['tags'] = movies['overview'].fillna('') + " " + movies['genres'].fillna('')
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['tags'].fillna(''))
    sim = cosine_similarity(tfidf_matrix)
    return movies, sim

movies, similarity = load_data()

# --- HERO SECTION ---
st.markdown('<p class="main-title">🎬 Movie Lounge</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-Powered Recommendations for Your Next Movie Night</p>', unsafe_allow_html=True)

# --- SEARCH & RECOMMEND (Centered) ---
col1, col2, col3 = st.columns([1,2,1])
with col2:
    selected_movie = st.selectbox("Search for a movie...", movies['title'].values)
    btn_col1, btn_col2, btn_col3 = st.columns([1,1,1])
    with btn_col2:
        recommend_btn = st.button('Recommend')

if recommend_btn:
    movie_indices = movies[movies['title'] == selected_movie].index
    if not movie_indices.empty:
        idx = movies.index.get_loc(movie_indices[0])
        distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
        
        st.write("---")
        st.subheader(f"Recommendations for {selected_movie}")
        cols = st.columns(5)
        for i in range(1, 6):
            movie_idx = distances[i][0]
            with cols[i-1]:
                st.image(fetch_poster(movies.iloc[movie_idx].id))
                st.caption(movies.iloc[movie_idx].title)

st.write("---")

# --- TRENDING & TOP RATED ---
def display_row(title, data_frame):
    st.subheader(title)
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(fetch_poster(data_frame.iloc[i].id))
            st.caption(data_frame.iloc[i].title)

display_row("🔥 Trending Now", movies.sort_values(by='popularity', ascending=False).head(5))
display_row("⭐ Top Rated by Critics", movies.sort_values(by='vote_average', ascending=False).head(5))
