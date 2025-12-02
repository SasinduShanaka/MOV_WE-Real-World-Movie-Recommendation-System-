# app.py — hybrid: title-based (content) + keyword/genre search fallback
from flask import Flask, render_template, request
import joblib
import pandas as pd
import difflib

app = Flask(__name__)

# 1) Load similarity artifacts (title-based recommender)
sim_matrix = joblib.load("model/sim_matrix.pkl")
indices = joblib.load("model/indices.pkl")     # title -> idx

# 2) Load movies with poster_url (from model)
movies_posters = joblib.load("model/movies.pkl")   # DataFrame with at least 'title','id','poster_url'

# 3) Load original TMDb CSV (has genres, keywords, overview)
meta = pd.read_csv("data/tmdb_5000_movies.csv")
# Keep titles consistent (strip)
meta['title'] = meta['title'].astype(str).str.strip()
# quick lowercase title list from meta (for searching)
meta['title_lc'] = meta['title'].str.lower()

# Build helper lists / maps
all_titles = movies_posters['title'].astype(str).tolist()
lower_to_title = {t.lower(): t for t in all_titles}

# If poster_url missing in some rows, ensure column exists
if 'poster_url' not in movies_posters.columns:
    if 'poster_path' in movies_posters.columns:
        movies_posters['poster_url'] = "https://image.tmdb.org/t/p/w342" + movies_posters['poster_path'].fillna('')
    else:
        movies_posters['poster_url'] = ""

def find_best_title(query):
    """Try exact / case-insensitive / fuzzy matching on titles."""
    if not query:
        return None, None
    q = query.strip()
    # exact
    if q in indices:
        return q, 'exact'
    # case-insensitive exact
    low = q.lower()
    if low in lower_to_title:
        return lower_to_title[low], 'case-insensitive'
    # difflib fuzzy on full titles
    matches = difflib.get_close_matches(q, all_titles, n=5, cutoff=0.65)
    if matches:
        return matches[0], 'fuzzy'
    # fuzzy on lowercase titles
    matches = difflib.get_close_matches(low, [t.lower() for t in all_titles], n=5, cutoff=0.65)
    if matches:
        cand_low = matches[0]
        return lower_to_title.get(cand_low, None), 'fuzzy-lower'
    return None, None

def title_recommend(best_title, topn=10):
    """Return top-n recommendations (title-based) as records with poster_url."""
    if best_title not in indices:
        return []
    idx = indices[best_title]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1: topn + 1]
    movie_indices = [i[0] for i in scores]
    # Defensive: movies_posters may not be in same order/index as sim matrix; assume indices are built from original movies DF
    # movies_posters was built from the same movies DataFrame, so iloc should work
    out = movies_posters.iloc[movie_indices][['title','poster_url','id']].to_dict(orient='records')
    return out

def keyword_search(q, topn=20):
    """Search genres/keywords/overview in the original CSV and return top matches with poster URL if available."""
    if not q:
        return []
    q = q.lower().strip()
    # Check raw text fields in meta (these are stringified JSON for genres/keywords, but contains the names)
    mask = (
        meta['genres'].astype(str).str.lower().str.contains(q, na=False) |
        meta['keywords'].astype(str).str.lower().str.contains(q, na=False) |
        meta['overview'].astype(str).str.lower().str.contains(q, na=False) |
        meta['title_lc'].str.contains(q, na=False)
    )
    matches = meta[mask].copy()
    if matches.empty:
        return []
    # prefer popular (if popularity exists), else just head
    if 'popularity' in matches.columns:
        matches = matches.sort_values(by='popularity', ascending=False)
    matches = matches.head(topn)
    # attach poster_url from movies_posters if available (join on title)
    merged = matches.merge(movies_posters[['title','poster_url','id']], on='title', how='left')
    # create list of dicts (title, poster_url, id)
    out = merged[['title','poster_url','id']].to_dict(orient='records')
    return out

@app.route("/", methods=["GET","POST"])
def home():
    results = []
    query = ""
    used_title = None
    method = None
    if request.method == "POST":
        query = request.form.get("movie", "").strip()
        # 1) try to interpret as a movie title (preferred)
        best_title, method = find_best_title(query)
        if best_title:
            # title-based recommendations
            results = title_recommend(best_title, topn=12)
            used_title = best_title
        else:
            # fallback to keyword/genre search
            results = keyword_search(query, topn=24)
            used_title = None
    # datalist suggestions: use top N titles for quicker loading
    titles_for_datalist = all_titles[:2000]
    return render_template("index.html",
                           results=results,
                           query=query,
                           used_title=used_title,
                           method=method,
                           titles=titles_for_datalist)

if __name__ == "__main__":
    app.run(debug=True)
