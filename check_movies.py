import joblib

movies = joblib.load("model/movies.pkl")
print("Columns:", movies.columns.tolist())
print(movies.head(10))
