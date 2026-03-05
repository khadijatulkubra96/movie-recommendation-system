import streamlit as st
import pickle
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Live Data Functions ---
def get_live_data(type="trending"):
    if type == "trending":
        url = "https://api.themoviedb.org/3/trending/movie/day?api_key=8265bd1679663a7ea12ac168da84d2e8"
    else:
        url = "https://api.themoviedb.org/3/movie/top_rated?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US&page=1"
        
    try:
        response = requests.get(url).json()
        return response['results'][:5] 
    except:
        return []

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    try:
        data = requests.get(url).json()
        return "https://image.tmdb.org/t/p/w500/" + data.get('poster_path', '')
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster"

st.set_page_config(page_title="Movie Lounge", layout="wide")
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), 
                    url('https://images.unsplash.com/photo-1536440136628-849c177e76a1?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
        background-attachment: fixed;
    }
    .main-title { font-size: 85px !important; font-weight: 800; text-align: center; color: #E50914; margin-top: -30px; }
    .sub-title { font-size: 28px !important; text-align: center; color: white; margin-bottom: 40px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_data():
    movies = pickle.load(open('movie_list.pkl', 'rb'))
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['tags'].fillna(''))
    sim = cosine_similarity(tfidf_matrix)
    return movies, sim

movies, similarity = load_data()

# --- Header ---
st.markdown('<p class="main-title">🎬 Movie Lounge</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-Powered Recommendations for Your Next Movie Night</p>', unsafe_allow_html=True)

# --- Search Section ---
col1, col2, col3 = st.columns([1,2,1])
with col2:
    selected_movie = st.selectbox("Search for a movie...", movies['title'].values)
    if st.button('Recommend'):
        movie_indices = movies[movies['title'] == selected_movie].index
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

# --- LIVE DYNAMIC SECTIONS ---

# 1. Live Trending Section
st.subheader("🔥 Trending Globally Today")
live_trending = get_live_data("trending")
if live_trending:
    t_cols = st.columns(5)
    for i, m in enumerate(live_trending):
        with t_cols[i]:
            st.image("https://image.tmdb.org/t/p/w500/" + m['poster_path'])
            st.caption(m['title'])

# 2. Live Top Rated Section
st.subheader("⭐ All-Time Classics (Live)")
live_top = get_live_data("top_rated")
if live_top:
    r_cols = st.columns(5)
    for i, m in enumerate(live_top):
        with r_cols[i]:
            st.image("https://image.tmdb.org/t/p/w500/" + m['poster_path'])
            st.caption(f"{m['title']} ({m['vote_average']})")
