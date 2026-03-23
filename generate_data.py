"""
generate_data.py
Fetches top anime from the Jikan API and generates synthetic user ratings.
Outputs: anime_info.csv, user_ratings.csv
"""

import requests
import pandas as pd
import numpy as np
import time
import random

API_URL = "https://api.jikan.moe/v4/top/anime"
MAX_PAGES = 40  # ~1,000 anime (sweet spot: fast + diverse)
NUM_USERS = 500
# Ratings per user will be set dynamically (5-15% of total anime)


def fetch_top_anime() -> list[dict]:
    """Fetch ALL top anime from Jikan API using auto-pagination."""
    all_anime = []
    page = 1
    max_retries = 3

    while page <= MAX_PAGES:
        print(f"\rFetching page {page}...", end=" ", flush=True)

        # Retry logic for rate-limit (429) errors
        success = False
        for attempt in range(max_retries):
            try:
                response = requests.get(API_URL, params={"page": page}, timeout=15)

                if response.status_code == 429:
                    wait = 2 * (attempt + 1)
                    print(f"⏳ Rate limited, waiting {wait}s...", end=" ", flush=True)
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                success = True
                break
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"\n⚠ Error on page {page}: {e}")
                    print(f"Stopping. Collected {len(all_anime)} anime so far.")
                    return all_anime

        if not success:
            print(f"\n⚠ Max retries reached on page {page}, stopping.")
            break

        data = response.json()

        for entry in data.get("data", []):
            title = entry.get("title_english") or entry.get("title", "Unknown")
            genres = ", ".join(g["name"] for g in entry.get("genres", []))
            all_anime.append({
                "anime_id": entry["mal_id"],
                "title": title,
                "image_url": entry.get("images", {}).get("jpg", {}).get("image_url", ""),
                "genres": genres if genres else "N/A",
                "score": entry.get("score", 0) or 0,
            })

        count = len(data.get("data", []))
        print(f"-> {count} anime (total: {len(all_anime)})")

        # Save progress every 50 pages
        if page % 50 == 0:
            _save_anime(all_anime)
            print(f"  💾 Progress saved ({len(all_anime)} anime)")

        # Check if there are more pages
        pagination = data.get("pagination", {})
        if not pagination.get("has_next_page", False):
            print("✓ Reached last page!")
            break

        page += 1
        time.sleep(1)

    return all_anime


def _save_anime(anime_list):
    """Helper to save anime list to CSV."""
    df = pd.DataFrame(anime_list).drop_duplicates(subset="anime_id")
    df.to_csv("anime_info.csv", index=False)
    return df


def generate_synthetic_ratings(anime_df: pd.DataFrame,
                                num_users: int = NUM_USERS) -> list[dict]:
    """
    Generate genre-aware synthetic user ratings.
    Each user has 2-4 favorite genres and rates anime accordingly:
      - Matching genres: higher scores (mean=8)
      - Non-matching genres: lower scores (mean=5)
    This creates realistic rating patterns so Cosine Similarity
    can find meaningful relationships between anime.
    """
    # Build genre lookup: anime_id -> set of genres
    genre_map = {}
    all_genres = set()
    for _, row in anime_df.iterrows():
        genres = set()
        raw = str(row.get("genres", ""))
        if raw and raw not in ("N/A", "nan"):
            genres = {g.strip() for g in raw.split(",") if g.strip()}
        genre_map[row["anime_id"]] = genres
        all_genres.update(genres)

    all_genres = list(all_genres) if all_genres else ["General"]
    anime_ids = anime_df["anime_id"].tolist()
    total = len(anime_ids)

    # Each user rates 5-15% of all anime (min 10, max total)
    min_r = max(10, int(total * 0.05))
    max_r = min(total, max(min_r, int(total * 0.15)))
    print(f"  Generating genre-aware ratings: {num_users} users × {min_r}-{max_r} ratings each...")
    print(f"  Found {len(all_genres)} unique genres")

    ratings = []
    for user_id in range(1, num_users + 1):
        # Each user has 2-4 favorite genres
        num_fav = random.randint(2, min(4, len(all_genres)))
        fav_genres = set(random.sample(all_genres, num_fav))

        num_ratings = random.randint(min_r, max_r)

        # Bias selection: 70% from favorite genres, 30% random
        fav_anime = [aid for aid in anime_ids if genre_map.get(aid, set()) & fav_genres]
        other_anime = [aid for aid in anime_ids if aid not in fav_anime]

        num_fav_picks = min(int(num_ratings * 0.7), len(fav_anime))
        num_other_picks = min(num_ratings - num_fav_picks, len(other_anime))

        picked = set()
        if fav_anime and num_fav_picks > 0:
            picked.update(random.sample(fav_anime, num_fav_picks))
        if other_anime and num_other_picks > 0:
            picked.update(random.sample(other_anime, num_other_picks))

        for aid in picked:
            anime_genres = genre_map.get(aid, set())
            if anime_genres & fav_genres:
                # Anime matches user's favorite genres → high score
                rating = int(np.random.normal(loc=8, scale=1.2))
            else:
                # Anime doesn't match → lower score
                rating = int(np.random.normal(loc=5, scale=1.5))
            rating = min(10, max(1, rating))
            ratings.append({"user_id": user_id, "anime_id": aid, "rating": rating})

    return ratings


def main():
    print("=" * 50)
    print("  Anime Data Generator")
    print("=" * 50)

    anime_list = fetch_top_anime()
    if not anime_list:
        print("ERROR: No anime fetched.")
        return

    anime_df = _save_anime(anime_list)
    print(f"\n✓ Saved {len(anime_df)} anime to anime_info.csv")

    ratings = generate_synthetic_ratings(anime_df)
    ratings_df = pd.DataFrame(ratings)
    ratings_df.to_csv("user_ratings.csv", index=False)
    print(f"✓ Saved {len(ratings_df)} ratings from {NUM_USERS} users to user_ratings.csv")
    print("\nDone!")


if __name__ == "__main__":
    main()
