
# 🎬 Movie Lounge: Hybrid Recommendation System

A professional Machine Learning web application that suggests movies based on content similarity and popularity.

##  Features
* **Hybrid Engine:** Combines NLP (TF-IDF) on movie overviews with statistical weights (Popularity/Ratings).
* **Live Posters:** Integrated with TMDb API to fetch real-time movie covers.
* **Modern UI:** Built with Streamlit, featuring a responsive dark-themed grid layout.
* **Statistical Validation:** Backed by ANOVA testing to ensure genre-based recommendation accuracy.

##  Tech Stack
- **Language:** Python 3.12
- **ML/NLP:** Scikit-learn (TF-IDF, Cosine Similarity)
- **Data:** Pandas, NumPy
- **Visuals:** Seaborn, Matplotlib
- **Web:** Streamlit, Requests (API Handling)

##  How it Works
1. The user selects a movie from the database.
2. The system calculates similarity using a pre-computed Cosine Similarity matrix.
3. It filters the top matches and ranks them using a weighted hybrid score.
4. Real-time posters are fetched via TMDb API for a premium user experience.
