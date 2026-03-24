"""
nn_recommender.py
Neural Network (Autoencoder) based Anime Recommender.
Uses PyTorch to learn latent user preference patterns from the User-Item Matrix.

Architecture:
    Input(N) → 128 → 32 (Latent Space) → 128 → Output(N)
    where N = number of anime

Features:
    - Train/Test Split (80/20)
    - Evaluation Metrics: RMSE, Precision@K
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


# ── Train/Test Split ────────────────────────────────────────────────────────
def split_train_test(user_item: np.ndarray, test_ratio: float = 0.2, seed: int = 42):
    """
    แบ่งข้อมูลเป็น Train/Test โดย mask rating บางส่วนออกจาก Train.

    วิธีการ: สำหรับแต่ละ User สุ่มซ่อน 20% ของ rating ที่มีอยู่
    แล้วเก็บไว้ใน Test set เพื่อวัดผลทีหลัง

    Args:
        user_item: User-Item Matrix (normalized 0-1)
        test_ratio: สัดส่วน Test set (default 20%)
        seed: Random seed สำหรับ reproducibility

    Returns:
        (train_matrix, test_matrix)
        - train_matrix: Rating ที่ใช้เทรน (80%)
        - test_matrix: Rating ที่ซ่อนไว้วัดผล (20%)
    """
    rng = np.random.RandomState(seed)
    train_matrix = user_item.copy()
    test_matrix = np.zeros_like(user_item)

    for i in range(user_item.shape[0]):
        # หา index ของ rating ที่ไม่ใช่ 0
        rated_indices = np.where(user_item[i] > 0)[0]
        if len(rated_indices) < 2:
            continue

        # สุ่มเลือก 20% ไปเป็น Test
        n_test = max(1, int(len(rated_indices) * test_ratio))
        test_indices = rng.choice(rated_indices, size=n_test, replace=False)

        # ย้ายจาก Train ไป Test
        test_matrix[i, test_indices] = train_matrix[i, test_indices]
        train_matrix[i, test_indices] = 0  # ซ่อนจาก Train

    return train_matrix, test_matrix


# ── Evaluation Metrics ──────────────────────────────────────────────────────
def compute_rmse(model, test_matrix: np.ndarray, train_matrix: np.ndarray) -> float:
    """
    คำนวณ RMSE (Root Mean Squared Error) บน Test set.

    RMSE = sqrt(mean((predicted - actual)^2))
    ยิ่งต่ำยิ่งดี — หมายความว่าโมเดลทำนายแม่นยำ
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(train_matrix)
        predictions = model(input_tensor).numpy()

    # เอาเฉพาะตำแหน่งที่มี rating ใน Test set
    mask = test_matrix > 0
    if mask.sum() == 0:
        return 0.0

    predicted = predictions[mask]
    actual = test_matrix[mask]

    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    return float(rmse)


def compute_precision_at_k(
    model, test_matrix: np.ndarray, train_matrix: np.ndarray, k: int = 5, threshold: float = 0.7
) -> float:
    """
    คำนวณ Precision@K — ใน K เรื่องที่แนะนำ มีกี่เรื่องที่ User จริงๆ ชอบ?

    Precision@K = (จำนวนเรื่องที่แนะนำถูก) / K

    threshold: rating >= 0.7 (= 7/10) ถือว่า "ชอบ"
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(train_matrix)
        predictions = model(input_tensor).numpy()

    precisions = []

    for i in range(test_matrix.shape[0]):
        # หาอนิเมะที่ User ชอบจริงๆ ใน Test set (rating >= threshold)
        actual_liked = set(np.where(test_matrix[i] >= threshold)[0])
        if not actual_liked:
            continue

        # หาอนิเมะที่โมเดลแนะนำ Top-K (จากที่ไม่มีใน Train)
        train_mask = train_matrix[i] == 0  # เฉพาะที่ไม่เคยเห็น
        pred_scores = predictions[i].copy()
        pred_scores[~train_mask] = -1  # ไม่พิจารณาเรื่องที่เคยเห็นแล้ว
        top_k_indices = set(np.argsort(pred_scores)[-k:])

        # Precision = ตัดกัน / K
        hits = len(actual_liked & top_k_indices)
        precisions.append(hits / k)

    return float(np.mean(precisions)) if precisions else 0.0


# ── Training Function ──────────────────────────────────────────────────────
def train_autoencoder(
    ratings_df: pd.DataFrame,
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 32,
) -> tuple:
    """
    ฝึก Autoencoder จาก User-Item Matrix พร้อม Train/Test Split.

    Returns:
        (model, user_item_matrix, train_loss_history, test_loss_history, eval_metrics, anime_ids)

        eval_metrics = {
            "rmse": float,
            "precision_at_5": float,
            "train_size": int,
            "test_size": int,
            "test_ratio": float,
        }
    """
    # ── สร้าง User-Item Matrix ──────────────────────────────────────────
    user_item = ratings_df.pivot_table(
        index="user_id", columns="anime_id", values="rating"
    ).fillna(0)

    anime_ids = user_item.columns.tolist()
    input_dim = len(anime_ids)

    # Normalize ratings to 0-1 range (ratings are 1-10)
    matrix = user_item.values.astype(np.float32) / 10.0

    # ── Train/Test Split (80/20) ────────────────────────────────────────
    train_matrix, test_matrix = split_train_test(matrix, test_ratio=0.2)

    train_ratings_count = int((train_matrix > 0).sum())
    test_ratings_count = int((test_matrix > 0).sum())
    total_ratings_count = train_ratings_count + test_ratings_count

    # ── สร้าง DataLoader (ใช้เฉพาะ Train data) ─────────────────────────
    train_tensor = torch.FloatTensor(train_matrix)
    dataset = TensorDataset(train_tensor, train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ── สร้างและฝึกโมเดล ────────────────────────────────────────────────
    model = AnimeAutoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_history = []
    test_loss_history = []

    test_tensor = torch.FloatTensor(test_matrix)
    test_mask = (test_tensor > 0).float()

    for epoch in range(epochs):
        # ── Train ────────────────────────────────────────────────────
        model.train()
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

        avg_train_loss = epoch_loss / max(num_batches, 1)
        train_loss_history.append(avg_train_loss)

        # ── Test Loss (ทุก epoch) ────────────────────────────────────
        model.eval()
        with torch.no_grad():
            test_output = model(train_tensor)  # ป้อน Train data แต่วัดผลบน Test
            test_loss = criterion(test_output * test_mask, test_tensor * test_mask)
            test_loss_history.append(test_loss.item())

    # ── คำนวณ Evaluation Metrics ────────────────────────────────────────
    model.eval()
    rmse = compute_rmse(model, test_matrix, train_matrix)
    precision_at_5 = compute_precision_at_k(model, test_matrix, train_matrix, k=5)

    eval_metrics = {
        "rmse": round(rmse, 4),
        "precision_at_5": round(precision_at_5, 4),
        "train_size": train_ratings_count,
        "test_size": test_ratings_count,
        "test_ratio": round(test_ratings_count / max(total_ratings_count, 1), 2),
    }

    return model, user_item, train_loss_history, test_loss_history, eval_metrics, anime_ids


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
    """
    # Find anime_id
    match = anime_df[anime_df["title"] == anime_title]
    if match.empty:
        return []

    anime_id = match.iloc[0]["anime_id"]
    anime_ids = user_item.columns.tolist()

    if anime_id not in anime_ids:
        return []

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

    # แสดง Predicted Rating เป็น % (จาก 1-10 scale)
    results = []
    for aid, score in top_scores.items():
        info = anime_df[anime_df["anime_id"] == aid]
        if not info.empty:
            pct = round(score * 10, 1)  # Predicted rating → percentage (ค่าจริง)
            pct = min(100.0, max(0.0, pct))
            results.append({
                "title": info.iloc[0]["title"],
                "match_percentage": pct,
                "image_url": info.iloc[0]["image_url"],
            })

    return results


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from recommender import load_data

    print("Loading data...")
    anime_df, ratings_df = load_data()

    print("Training Autoencoder (50 epochs) with Train/Test Split...")
    model, user_item, train_losses, test_losses, metrics, anime_ids = train_autoencoder(
        ratings_df, epochs=50
    )

    print(f"\n📊 Evaluation Metrics:")
    print(f"  Train size: {metrics['train_size']} ratings")
    print(f"  Test size:  {metrics['test_size']} ratings ({metrics['test_ratio']*100:.0f}%)")
    print(f"  Final Train Loss: {train_losses[-1]:.6f}")
    print(f"  Final Test Loss:  {test_losses[-1]:.6f}")
    print(f"  RMSE:          {metrics['rmse']:.4f}")
    print(f"  Precision@5:   {metrics['precision_at_5']:.4f} ({metrics['precision_at_5']*100:.1f}%)")

    sample_title = anime_df.iloc[0]["title"]
    print(f"\nNN Recommendations for '{sample_title}':")
    for rec in get_nn_recommendations(sample_title, anime_df, model, user_item):
        print(f"  {rec['title']} — {rec['match_percentage']}%")
