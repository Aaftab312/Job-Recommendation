import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

PHRASES = [
    "machine learning",
    "deep learning",
    "data science",
    "data analysis",
    "data structures",
    "natural language processing",
    "computer vision",
    "big data",
    "data visualization",
    "power bi"
]

df = pd.read_csv("jobs_dataset_1000.csv")

df["job_title"] = df["job_title"].fillna("").astype(str)
df["skills_required"] = df["skills_required"].fillna("").astype(str)
df["description"] = df["description"].fillna("").astype(str)

df = df.drop_duplicates(subset=["job_title", "skills_required", "description"]).reset_index(drop=True)
df["text"] = df["skills_required"] + " " + df["description"]

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["text"])

model_data = {
    "tfidf": tfidf,
    "tfidf_matrix": tfidf_matrix,
    "df": df,
    "phrases": PHRASES
}

joblib.dump(model_data, "model.pkl")
print("model.pkl created successfully")