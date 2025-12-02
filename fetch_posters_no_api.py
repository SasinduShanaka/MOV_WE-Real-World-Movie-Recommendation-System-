# fetch_posters_no_api.py
import joblib
import pandas as pd
import requests
import time
import os
from bs4 import BeautifulSoup

MOVIES_PKL = "model/movies.pkl"        # file produced by build_data.py
OUT_CSV = "model/movies_with_posters.csv"
SLEEP_SEC = 0.35                       # polite delay between requests

if not os.path.exists(MOVIES_PKL):
    raise SystemExit("model/movies.pkl not found. Run build_data.py first.")

movies = joblib.load(MOVIES_PKL)  # expects a DataFrame with at least 'id' and 'title'
# if movies is a Series or different structure adjust accordingly
if isinstance(movies, pd.Series):
    movies = movies.to_frame().T

# ensure columns exist
if 'id' not in movies.columns:
    raise SystemExit("No 'id' column in model/movies.pkl. The build step must have saved movie IDs.")

# Add poster_url column if missing
if 'poster_url' not in movies.columns:
    movies['poster_url'] = ""

base_movie_url = "https://www.themoviedb.org/movie/{}"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119 Safari/537.36"
}

updated = 0
total = len(movies)
print(f"Total movies: {total}")

for idx, row in movies.iterrows():
    try:
        # skip if already filled
        if row.get('poster_url'):
            continue

        tmdb_id = row.get('id')
        if pd.isna(tmdb_id) or tmdb_id == "":
            continue

        url = base_movie_url.format(int(tmdb_id))
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            print(f"[WARN] {row.get('title')} (id={tmdb_id}) status={resp.status_code}")
            time.sleep(SLEEP_SEC)
            continue

        soup = BeautifulSoup(resp.text, "lxml")

        # TMDb uses meta property="og:image" with full poster url
        og = soup.find("meta", property="og:image")
        poster_full = ""
        if og and og.get("content"):
            poster_full = og.get("content").strip()

        # fallback: search for <img class="poster" ...>
        if not poster_full:
            img = soup.select_one("img.poster")
            if img and img.get("src"):
                poster_full = img.get("src").strip()

        # final fallback: try finding the first image with 'poster' in src
        if not poster_full:
            imgs = soup.find_all("img")
            for i in imgs:
                src = i.get("src") or ""
                if "poster" in src or "/t/p/" in src:
                    poster_full = src
                    break

        if poster_full:
            movies.at[idx, 'poster_url'] = poster_full
            updated += 1
            print(f"[OK] {row.get('title')} -> {poster_full}")
        else:
            print(f"[NOPOST] {row.get('title')} -> not found")

    except Exception as e:
        print(f"[ERR] {row.get('title')} error={e}")

    # polite delay
    time.sleep(SLEEP_SEC)

print(f"Done. Updated poster_url for {updated} movies out of {total}.")

# Save updated DataFrame
joblib.dump(movies, MOVIES_PKL)
movies.to_csv(OUT_CSV, index=False)
print(f"Saved updated movies to {MOVIES_PKL} and CSV {OUT_CSV}")
