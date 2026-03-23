"""
recommender.py
Item-based Collaborative Filtering using Cosine Similarity.
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    """Load anime info and user ratings from CSV files."""
    anime_df = pd.read_csv("anime_info.csv")
    ratings_df = pd.read_csv("user_ratings.csv")
    return anime_df, ratings_df


def build_item_similarity_matrix(ratings_df: pd.DataFrame):
    """Build a user-item matrix and compute item-item cosine similarity."""
    # Create the user-item pivot table (rows=anime, columns=users)
    user_item_matrix = ratings_df.pivot_table(
        index="anime_id",
        columns="user_id",
        values="rating",
    ).fillna(0)

    # Compute cosine similarity between items (anime)
    similarity_matrix = cosine_similarity(user_item_matrix)

    # Wrap in a DataFrame for easy lookup
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=user_item_matrix.index,
        columns=user_item_matrix.index,
    )
    return similarity_df


def get_recommendations(anime_title: str,
                         anime_df: pd.DataFrame = None,
                         similarity_df: pd.DataFrame = None,
                         top_n: int = 5) -> list[dict]:
    """
    Get top-N similar anime for a given title.

    Returns a list of dicts with keys: title, match_percentage, image_url
    """
    if anime_df is None or similarity_df is None:
        anime_df, ratings_df = load_data()
        similarity_df = build_item_similarity_matrix(ratings_df)

    # Find the anime_id for the given title
    match = anime_df[anime_df["title"] == anime_title]
    if match.empty:
        return []

    anime_id = match.iloc[0]["anime_id"]

    if anime_id not in similarity_df.index:
        return []

    # Get similarity scores, drop self, sort descending
    sim_scores = similarity_df[anime_id].drop(anime_id, errors="ignore")
    sim_scores = sim_scores.sort_values(ascending=False).head(top_n)

    # Build results
    results = []
    for similar_id, score in sim_scores.items():
        info = anime_df[anime_df["anime_id"] == similar_id]
        if not info.empty:
            results.append({
                "title": info.iloc[0]["title"],
                "match_percentage": round(score * 100, 1),
                "image_url": info.iloc[0]["image_url"],
            })

    return results


# Quick smoke test
if __name__ == "__main__":
    anime_df, ratings_df = load_data()
    sim_df = build_item_similarity_matrix(ratings_df)
    sample_title = anime_df.iloc[0]["title"]
    print(f"Recommendations for '{sample_title}':")
    for rec in get_recommendations(sample_title, anime_df, sim_df):
        print(f"  {rec['title']} — {rec['match_percentage']}%")
