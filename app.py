import streamlit as st
import pickle
import requests

# --- Helper Functions ---
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    try:
        data = requests.get(url).json()
        return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster"

# --- Page Layout ---
st.set_page_config(page_title="Movie Lounge", layout="wide")

# Custom Styling
st.markdown("<style>.stApp {background-color: #0e1117; color: white;}</style>", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    # File load
    movies = pickle.load(open('movie_list.pkl', 'rb'))
    
    if 'tags' not in movies.columns:
        st.error("Column 'tags' not found in the pickle file. Please re-upload the correct movie_list.pkl!")
        movies['tags'] = movies['overview'].fillna('') + " " + movies['genres'].fillna('')
        
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['tags'].fillna(''))
    sim = cosine_similarity(tfidf_matrix)
    return movies, sim
movies, similarity = load_data()

# --- SECTION 1: SEARCH & RECOMMEND ---
selected_movie = st.selectbox("Search for a movie...", movies['title'].values)

if st.button('Recommend'):
    idx = movies[movies['title_clean'] == selected_movie.lower().strip()].index[0]
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

# --- SECTION 2: TRENDING NOW (Popularity Based) ---
st.subheader("🔥 Trending Now")
trending = movies.sort_values(by='popularity', ascending=False).head(5)
cols_trend = st.columns(5)
for i in range(5):
    with cols_trend[i]:
        st.image(fetch_poster(trending.iloc[i].id))
        st.caption(trending.iloc[i].title)

# --- SECTION 3: TOP CRITIC'S CHOICE (Rating Based) ---
st.subheader("⭐ Top Rated by Critics")
top_rated = movies.sort_values(by='vote_average', ascending=False).head(5)
cols_rate = st.columns(5)
for i in range(5):
    with cols_rate[i]:
        st.image(fetch_poster(top_rated.iloc[i].id))
        st.caption(f"{top_rated.iloc[i].title} ({top_rated.iloc[i].vote_average})")
