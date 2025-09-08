import pandas as pd
import numpy as np
import ast
import pickle
from sentence_transformers import SentenceTransformer

# Load datasets
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

# Merge datasets
movies = movies.merge(credits, on='title')
movies = movies[['title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

# Functions to clean JSON fields
def parse_names(text):
    try:
        return [i['name'].replace(" ", "") for i in ast.literal_eval(text)]
    except:
        return []

def top_cast(text, k=3):
    try:
        return [i['name'].replace(" ", "") for i in ast.literal_eval(text)[:k]]
    except:
        return []

def fetch_director(text):
    try:
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                return [i['name'].replace(" ", "")]
        return []
    except:
        return []

# Apply
movies['genres'] = movies['genres'].apply(parse_names)
movies['keywords'] = movies['keywords'].apply(parse_names)
movies['cast'] = movies['cast'].apply(top_cast)
movies['crew'] = movies['crew'].apply(fetch_director)

# Combine into one text field
def combine_features(row):
    return " ".join(row['overview'].split()) + " " + \
           " ".join(row['genres']) + " " + \
           " ".join(row['keywords']) + " " + \
           " ".join(row['cast']) + " " + \
           " ".join(row['crew'])

movies['combined'] = movies.apply(combine_features, axis=1)

# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Encoding movie embeddings... ⏳")
embeddings = model.encode(movies['combined'].tolist(), show_progress_bar=True)

# Save movies and embeddings
movies.to_pickle("models/movies.pkl")
with open("models/embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("✅ Data prepared and saved!")

