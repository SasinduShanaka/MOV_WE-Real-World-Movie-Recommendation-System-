# app.py
from flask import Flask, render_template, request
import joblib
import pandas as pd
import difflib
import os

app = Flask(__name__)

# ---------- Paths ----------
MODEL_DIR = "model"
SIM_MATRIX_PKL = os.path.join(MODEL_DIR, "sim_matrix.pkl")
INDICES_PKL = os.path.join(MODEL_DIR, "indices.pkl")
MOVIES_PKL = os.path.join(MODEL_DIR, "movies.pkl")
TMDB_CSV = os.path.join("data", "tmdb_5000_movies.csv")

for path in (SIM_MATRIX_PKL, INDICES_PKL, MOVIES_PKL, TMDB_CSV):
    if not os.path.exists(path):
        raise SystemExit(f"Required file not found: {path}. Run earlier steps (build_data / scraping) first.")

# ---------- Load artifacts ----------
sim_matrix = joblib.load(SIM_MATRIX_PKL)
indices = joblib.load(INDICES_PKL)      # Series: title -> idx
movies_posters = joblib.load(MOVIES_PKL)  # DataFrame with title,id,poster_url (or poster_path)
meta = pd.read_csv(TMDB_CSV)

# ---------- Normalize / prepare metadata ----------
meta['title'] = meta['title'].astype(str).str.strip()
meta['title_lc'] = meta['title'].str.lower()

# Ensure poster_url exists in movies_posters
if 'poster_url' not in movies_posters.columns:
    if 'poster_path' in movies_posters.columns:
        movies_posters['poster_url'] = "https://image.tmdb.org/t/p/w342" + movies_posters['poster_path'].fillna('')
    else:
        movies_posters['poster_url'] = ""

# Merge small meta info into posters DF for easier display
meta_small = meta[['title', 'overview', 'vote_average', 'release_date', 'id']].copy()
meta_small = meta_small.drop_duplicates(subset=['title'], keep='first')
movies_full = movies_posters.merge(meta_small, on='title', how='left', suffixes=('','_meta'))

# ---------- Helper structures ----------
all_titles = movies_posters['title'].astype(str).tolist()
lower_to_title = {t.lower(): t for t in all_titles}

GENRES = {
    "action","horror","romance","thriller","crime","comedy","adventure",
    "animation","drama","fantasy","sci-fi","scifi","family","mystery",
    "war","western","documentary","musical","biography","history"
}

# ---------- Matching & recommendation helpers ----------
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

def format_movie_record(row):
    """Return dict with fields for the template."""
    title = row.get('title')
    poster_url = row.get('poster_url') or ""
    tmdb_id = row.get('id') if 'id' in row else None
    if (tmdb_id is None or pd.isna(tmdb_id)) and 'id_meta' in row:
        tmdb_id = row.get('id_meta')
    tmdb_link = ""
    try:
        if tmdb_id not in (None, "", float('nan')):
            tmdb_link = f"https://www.themoviedb.org/movie/{int(tmdb_id)}"
    except Exception:
        tmdb_link = ""
    overview = row.get('overview') or ""
    short_overview = (overview[:200].rsplit(' ', 1)[0] + '…') if overview and len(overview) > 200 else overview
    rating = None
    if 'vote_average' in row:
        try:
            rating = round(float(row.get('vote_average')), 1)
        except Exception:
            rating = None
    elif 'vote_average_meta' in row:
        try:
            rating = round(float(row.get('vote_average_meta')), 1)
        except Exception:
            rating = None
    release_date = row.get('release_date') or row.get('release_date_meta') or ""
    year = ""
    if isinstance(release_date, str) and release_date.strip():
        year = release_date.split('-')[0]
    return {
        "title": title,
        "poster_url": poster_url,
        "tmdb_link": tmdb_link,
        "overview": short_overview,
        "rating": rating,
        "year": year,
        "id": int(tmdb_id) if (tmdb_id not in (None, "", float('nan'))) else None
    }

def title_recommend(best_title, topn=10):
    """Return top-n content-based recommendations for a given title."""
    if best_title not in indices:
        return []
    idx = indices[best_title]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1: topn + 1]   # skip itself
    movie_indices = [i[0] for i in scores]
    rows = movies_full.reset_index().iloc[movie_indices]
    records = [format_movie_record(r) for _, r in rows.iterrows()]
    return records

# ---------- FIXED keyword_search (Option A) ----------
def keyword_search(q, topn=20):
    """Search meta fields (genres, keywords, overview, title) and return results."""
    if not q:
        return []
    q = q.lower().strip()
    mask = (
        meta['genres'].astype(str).str.lower().str.contains(q, na=False) |
        meta['keywords'].astype(str).str.lower().str.contains(q, na=False) |
        meta['overview'].astype(str).str.lower().str.contains(q, na=False) |
        meta['title_lc'].str.contains(q, na=False)
    )
    matches = meta[mask].copy()
    if matches.empty:
        return []
    if 'popularity' in matches.columns:
        matches = matches.sort_values(by='popularity', ascending=False)
    matches = matches.head(topn)

    # Merge but keep movies_full columns 'overview', 'vote_average', 'release_date' by using suffixes
    merged = matches.merge(
        movies_full[['title','poster_url','id','overview','vote_average','release_date']],
        on='title',
        how='left',
        suffixes=('_meta','')   # left gets _meta, right keeps original names
    )

    # Now merged should have 'overview','vote_average','release_date' (from movies_full when available)
    records = [format_movie_record(r) for _, r in merged.iterrows()]
    return records

# ---------- Routes ----------
@app.route("/", methods=["GET","POST"])
def home():
    results = []
    query = ""
    used_title = None
    method = None
    if request.method == "POST":
        query = request.form.get("movie", "").strip()
        q_lower = query.lower()

        # If query looks exactly like a genre, prefer keyword search
        if q_lower in GENRES:
            results = keyword_search(query, topn=24)
            used_title = None
            method = 'genre-fallback'
        else:
            best_title, method = find_best_title(query)
            if best_title:
                results = title_recommend(best_title, topn=12)
                used_title = best_title
            else:
                results = keyword_search(query, topn=24)
                used_title = None

    titles_for_datalist = all_titles[:2000]  # limit for performance
    return render_template("index.html",
                           results=results,
                           query=query,
                           used_title=used_title,
                           method=method,
                           titles=titles_for_datalist)

# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True)
