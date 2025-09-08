from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load saved data
movies = pd.read_pickle("models/movies.pkl")
with open("models/embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# Precompute similarity
similarity = cosine_similarity(embeddings)

# Recommendation function
def recommend(movie, top_k=5):
    if movie not in movies['title'].values:
        return []
    idx = movies[movies['title'] == movie].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_k+1]
    return [movies.iloc[i]['title'] for i, score in sim_scores]

@app.route("/", methods=["GET", "POST"])
def home():
    recs = []
    movie_name = ""
    error = ""
    if request.method == "POST":
        movie_name = request.form["movie"].strip()
        recs = recommend(movie_name)
        if not recs:
            error = f"❌ Movie '{movie_name}' not found in dataset."
    return render_template("index.html", recs=recs, selected=movie_name, error=error)

if __name__ == "__main__":
    app.run(debug=True)
