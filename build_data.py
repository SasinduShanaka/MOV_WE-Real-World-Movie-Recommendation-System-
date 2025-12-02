import pandas as pd
import numpy as np
import ast
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load datasets
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

# Merge movies + credits
movies = movies.merge(credits, on="title")

# Helper functions
def parse(text):
    try:
        return ast.literal_eval(text)
    except:
        return []

def get_names(arr):
    return [i['name'].replace(" ", "") for i in arr if 'name' in i]

def get_cast(arr):
    return [i['name'].replace(" ", "") for i in arr[:5] if 'name' in i]

def get_director(arr):
    for i in arr:
        if i.get("job") == "Director":
            return i.get("name").replace(" ", "")
    return ""

# Extract features
movies["genres_list"] = movies["genres"].apply(parse).apply(get_names)
movies["keywords_list"] = movies["keywords"].apply(parse).apply(get_names)
movies["cast_list"] = movies["cast"].apply(parse).apply(get_cast)
movies["crew_parsed"] = movies["crew"].apply(parse)
movies["director"] = movies["crew_parsed"].apply(get_director)
movies["overview"] = movies["overview"].fillna("")

# Build a combined text field (soup)
def make_soup(row):
    return (
        row["overview"] + " " +
        " ".join(row["genres_list"]) + " " +
        " ".join(row["keywords_list"]) + " " +
        " ".join(row["cast_list"]) + " " +
        row["director"]
    )

movies["soup"] = movies.apply(make_soup, axis=1)

# TF-IDF for overview text
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(movies["overview"])

# CountVectorizer for metadata soup
count = CountVectorizer(stop_words="english", max_features=5000)
count_matrix = count.fit_transform(movies["soup"])

# Compute similarity
sim_overview = cosine_similarity(tfidf_matrix)
sim_meta = cosine_similarity(count_matrix)

# Weighted similarity matrix
SIM = 0.4 * sim_overview + 0.6 * sim_meta

# Build index (title -> index)
movies = movies.reset_index()
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

# Save model data
os.makedirs("model", exist_ok=True)

# ensure poster_path exists
if 'poster_path' not in movies.columns:
    print("poster_path column missing — creating empty column")
    movies['poster_path'] = ""

# then save only the safe subset
cols = [c for c in ["title", "poster_path", "id"] if c in movies.columns]
joblib.dump(movies[cols], "model/movies.pkl")


joblib.dump(SIM, "model/sim_matrix.pkl")
joblib.dump(indices, "model/indices.pkl")
joblib.dump(movies[["title", "poster_path", "id"]], "model/movies.pkl")

print("✔ Similarity matrix and model files saved!")
