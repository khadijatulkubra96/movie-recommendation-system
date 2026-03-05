import streamlit as st
import pickle
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIG and CSS ---
st.set_page_config(page_title="Movie Lounge", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    /* Hero Section Background */
    .hero-container {
        background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), url('https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
        background-position: center;
        padding: 80px 20px;
        text-align: center;
        border-radius: 15px;
        margin-bottom: 40px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    .hero-title { color: #E50914; font-size: 60px; font-weight: 800; text-transform: uppercase; margin: 0; }
    .hero-tagline { color: #f0f0f0; font-size: 22px; margin-top: 10px; }
    /* Section Headings */
    .section-header { border-left: 5px solid #E50914; padding-left: 15px; margin-top: 30px; margin-bottom: 20px; font-size: 24px; font-weight: bold; }
    </style>
    <div class="hero-container">
        <h1 class="hero-title">Movie Lounge</h1>
        <p class="hero-tagline">AI-Powered Recommendations for Your Next Movie Night</p>
    </div>
    """, unsafe_allow_html=True)

# --- 2. DATA LOADING ---
@st.cache_resource
def load_data():
    movies = pickle.load(open('movie_list.pkl', 'rb'))
    # NLP Matrix Calculation
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['tags'])
    sim = cosine_similarity(tfidf_matrix)
    return movies, sim

movies, similarity = load_data()

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    try:
        data = requests.get(url).json()
        return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster"

# --- 3. SEARCH & RECOMMENDATION ---
st.markdown('<div class="section-header">🔍 Find Your Next Watch</div>', unsafe_allow_html=True)
selected_movie = st.selectbox("", movies['title'].values)

if st.button('Get Recommendations'):
    idx = movies[movies['title'] == selected_movie].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    
    st.markdown(f"### Top Matches for **{selected_movie}**")
    cols = st.columns(5)
    for i in range(1, 6):
        movie_idx = distances[i][0]
        with cols[i-1]:
            st.image(fetch_poster(movies.iloc[movie_idx].id))
            st.markdown(f"**{movies.iloc[movie_idx].title}**")
            st.caption(f"⭐ Rating: {movies.iloc[movie_idx].vote_average}/10")

# --- 4. DEFAULT SECTIONS 
st.markdown('<div class="section-header">🔥 Trending This Week</div>', unsafe_allow_html=True)
trending = movies.sort_values(by='popularity', ascending=False).head(5)
t_cols = st.columns(5)
for i in range(5):
    with t_cols[i]:
        st.image(fetch_poster(trending.iloc[i].id))
        st.caption(f"**{trending.iloc[i].title}**")

st.markdown('<div class="section-header">⭐ All-Time Hits</div>', unsafe_allow_html=True)
top_rated = movies.sort_values(by='vote_average', ascending=False).head(5)
r_cols = st.columns(5)
for i in range(5):
    with r_cols[i]:
        st.image(fetch_poster(top_rated.iloc[i].id))
        st.caption(f"**{top_rated.iloc[i].title}** ({top_rated.iloc[i].vote_average})")
