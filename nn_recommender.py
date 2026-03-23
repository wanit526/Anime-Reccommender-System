"""
nn_recommender.py
Neural Network (Autoencoder) based Anime Recommender.
Uses PyTorch to learn latent user preference patterns from the User-Item Matrix.

Architecture:
    Input(N) → 128 → 32 (Latent Space) → 128 → Output(N)
    where N = number of anime
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class AnimeAutoencoder(nn.Module):
    """
    Autoencoder สำหรับระบบแนะนำอนิเมะ
    Encoder: บีบข้อมูล Rating จาก N มิติ → 32 มิติ (Latent Space)
    Decoder: คืนกลับจาก 32 มิติ → N มิติ (Predicted Ratings)
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, input_dim),
            nn.Sigmoid(),  # Output 0-1 range (we'll scale ratings to 0-1)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def get_latent(self, x):
        """Extract latent features (32-dim) for a given input."""
        with torch.no_grad():
            return self.encoder(x)


def train_autoencoder(
    ratings_df: pd.DataFrame,
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 32,
) -> tuple:
    """
    ฝึก Autoencoder จาก User-Item Matrix.

    Args:
        ratings_df: DataFrame with columns [user_id, anime_id, rating]
        epochs: จำนวนรอบการฝึก
        lr: Learning rate
        batch_size: Batch size

    Returns:
        (model, user_item_matrix, loss_history, anime_ids)
    """
    # ── สร้าง User-Item Matrix ──────────────────────────────────────────
    user_item = ratings_df.pivot_table(
        index="user_id", columns="anime_id", values="rating"
    ).fillna(0)

    anime_ids = user_item.columns.tolist()
    input_dim = len(anime_ids)

    # Normalize ratings to 0-1 range (ratings are 1-10)
    matrix = user_item.values.astype(np.float32) / 10.0

    # ── สร้าง DataLoader ────────────────────────────────────────────────
    tensor_data = torch.FloatTensor(matrix)
    dataset = TensorDataset(tensor_data, tensor_data)  # input = target (autoencoder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ── สร้างและฝึกโมเดล ────────────────────────────────────────────────
    model = AnimeAutoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_input, batch_target in dataloader:
            optimizer.zero_grad()
            output = model(batch_input)

            # Only compute loss on non-zero ratings (mask)
            mask = (batch_target > 0).float()
            loss = criterion(output * mask, batch_target * mask)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        loss_history.append(avg_loss)

    model.eval()
    return model, user_item, loss_history, anime_ids


def get_nn_recommendations(
    anime_title: str,
    anime_df: pd.DataFrame,
    model: AnimeAutoencoder,
    user_item: pd.DataFrame,
    top_n: int = 5,
) -> list[dict]:
    """
    แนะนำอนิเมะโดยใช้ Neural Network.

    วิธีการ:
    1. หา Users ที่เคยให้คะแนนอนิเมะที่เลือก
    2. ใช้ Autoencoder ทำนาย Rating ที่ Users เหล่านั้นจะให้กับอนิเมะอื่นๆ
    3. หาค่าเฉลี่ยของ Predicted Rating แล้วเรียงจากมากไปน้อย

    Args:
        anime_title: ชื่ออนิเมะที่เลือก
        anime_df: DataFrame ข้อมูลอนิเมะ
        model: Trained Autoencoder model
        user_item: User-Item Matrix
        top_n: จำนวนที่แนะนำ

    Returns:
        list of dicts: [{title, match_percentage, image_url}, ...]
    """
    # Find anime_id
    match = anime_df[anime_df["title"] == anime_title]
    if match.empty:
        return []

    anime_id = match.iloc[0]["anime_id"]
    anime_ids = user_item.columns.tolist()

    if anime_id not in anime_ids:
        return []

    anime_idx = anime_ids.index(anime_id)

    # Find users who rated this anime
    users_who_rated = user_item[user_item[anime_id] > 0]
    if users_who_rated.empty:
        return []

    # Predict ratings using Autoencoder
    with torch.no_grad():
        input_tensor = torch.FloatTensor(users_who_rated.values.astype(np.float32) / 10.0)
        predictions = model(input_tensor).numpy() * 10.0  # Scale back to 1-10

    # Average predicted ratings across users
    avg_predictions = predictions.mean(axis=0)

    # Create score Series, exclude the selected anime
    scores = pd.Series(avg_predictions, index=anime_ids)
    scores = scores.drop(anime_id, errors="ignore")

    # Sort and get top-N
    top_scores = scores.sort_values(ascending=False).head(top_n)

    # Normalize scores to percentage (0-100)
    max_score = top_scores.max() if top_scores.max() > 0 else 1
    min_score = top_scores.min()
    score_range = max_score - min_score if max_score != min_score else 1

    results = []
    for aid, score in top_scores.items():
        info = anime_df[anime_df["anime_id"] == aid]
        if not info.empty:
            # Normalize to percentage
            pct = ((score - min_score) / score_range) * 30 + 70  # Map to 70-100%
            results.append({
                "title": info.iloc[0]["title"],
                "match_percentage": round(pct, 1),
                "image_url": info.iloc[0]["image_url"],
            })

    return results


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from recommender import load_data

    print("Loading data...")
    anime_df, ratings_df = load_data()

    print("Training Autoencoder (50 epochs)...")
    model, user_item, losses, anime_ids = train_autoencoder(ratings_df, epochs=50)
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Anime count: {len(anime_ids)}")

    sample_title = anime_df.iloc[0]["title"]
    print(f"\nNN Recommendations for '{sample_title}':")
    for rec in get_nn_recommendations(sample_title, anime_df, model, user_item):
        print(f"  {rec['title']} — {rec['match_percentage']}%")
