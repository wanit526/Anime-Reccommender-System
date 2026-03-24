"""
load_kaggle_data.py
โหลด Dataset จริงจาก Kaggle (Anime Recommendations Database)
แล้วแปลงให้อยู่ในรูปแบบที่ระบบ Recommender ใช้งานได้

วิธีใช้:
1. ไปดาวน์โหลด Dataset จาก:
   https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database
2. แตกไฟล์แล้ววาง anime.csv และ rating.csv ไว้ในโฟลเดอร์ kaggle_data/
3. รัน: python load_kaggle_data.py
4. ระบบจะสร้าง anime_info.csv + user_ratings.csv (ข้อมูลจริง!)
"""

import pandas as pd
import numpy as np
import requests
import time
import os

# ── ตั้งค่า ──────────────────────────────────────────────────────────────────
KAGGLE_DIR = "kaggle_data"                # โฟลเดอร์ที่เก็บไฟล์ Kaggle
ANIME_FILE = os.path.join(KAGGLE_DIR, "anime.csv")
RATING_FILE = os.path.join(KAGGLE_DIR, "rating.csv")

TOP_N_ANIME = 1000       # เลือกเฉพาะ Top 1,000 อนิเมะ (ยอดนิยมที่สุด)
TOP_N_USERS = 5000        # เลือกเฉพาะ 5,000 users ที่ให้คะแนนมากที่สุด
MIN_RATING = 1            # กรอง rating ที่ต่ำกว่า 1 ออก (-1 = "ดูแล้วแต่ไม่ให้คะแนน")

# URL สำหรับดึงรูปภาพจาก MAL
MAL_IMAGE_BASE = "https://cdn.myanimelist.net/images/anime"


def check_kaggle_files():
    """ตรวจว่ามีไฟล์ Kaggle อยู่ไหม"""
    if not os.path.exists(ANIME_FILE):
        print(f"❌ ไม่พบไฟล์ {ANIME_FILE}")
        print(f"\n📥 วิธีดาวน์โหลด:")
        print(f"   1. ไปที่ https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database")
        print(f"   2. คลิก Download → แตกไฟล์ zip")
        print(f"   3. สร้างโฟลเดอร์ '{KAGGLE_DIR}/' ในโปรเจกต์")
        print(f"   4. วาง anime.csv และ rating.csv ไว้ข้างใน")
        print(f"   5. รันสคริปต์นี้อีกครั้ง")
        return False
    if not os.path.exists(RATING_FILE):
        print(f"❌ ไม่พบไฟล์ {RATING_FILE}")
        return False
    return True


def load_and_process():
    """โหลด + ตัด + แปลง Dataset จาก Kaggle"""
    print("=" * 55)
    print("  📊 Kaggle Dataset Loader (Real Data)")
    print("=" * 55)

    # ── 1. โหลด Anime Data ──────────────────────────────────────────────
    print("\n1️⃣  โหลด anime.csv...")
    anime_raw = pd.read_csv(ANIME_FILE)
    print(f"   ข้อมูลอนิเมะทั้งหมด: {len(anime_raw)} เรื่อง")

    # ลบแถวที่ไม่มีชื่อหรือ genre
    anime_raw = anime_raw.dropna(subset=["name"])
    anime_raw["genre"] = anime_raw["genre"].fillna("N/A")
    anime_raw["rating"] = pd.to_numeric(anime_raw["rating"], errors="coerce").fillna(0)

    # เลือก Top N อนิเมะตาม members (จำนวนคนดู)
    anime_raw["members"] = pd.to_numeric(anime_raw["members"], errors="coerce").fillna(0)
    top_anime = anime_raw.nlargest(TOP_N_ANIME, "members")
    top_anime_ids = set(top_anime["anime_id"].tolist())
    print(f"   เลือก Top {TOP_N_ANIME} อนิเมะ (ตาม members)")

    # ── 2. โหลด Rating Data ─────────────────────────────────────────────
    print("\n2️⃣  โหลด rating.csv... (อาจใช้เวลาสักครู่)")
    rating_raw = pd.read_csv(RATING_FILE)
    total_raw = len(rating_raw)
    print(f"   Rating ทั้งหมด: {total_raw:,} รายการ")

    # กรอง: เฉพาะ anime ที่อยู่ใน Top N + rating >= 1
    rating_filtered = rating_raw[
        (rating_raw["anime_id"].isin(top_anime_ids)) &
        (rating_raw["rating"] >= MIN_RATING)
    ].copy()
    print(f"   หลังกรอง (Top {TOP_N_ANIME} anime + rating ≥ {MIN_RATING}): {len(rating_filtered):,} รายการ")

    # เลือก Top N users ที่ให้คะแนนมากที่สุด
    user_counts = rating_filtered["user_id"].value_counts()
    top_users = user_counts.head(TOP_N_USERS).index.tolist()
    rating_final = rating_filtered[rating_filtered["user_id"].isin(top_users)].copy()
    print(f"   เลือก Top {TOP_N_USERS} users: {len(rating_final):,} ratings")

    # Re-index user_id ให้เริ่มจาก 1
    user_id_map = {old: new for new, old in enumerate(sorted(rating_final["user_id"].unique()), 1)}
    rating_final["user_id"] = rating_final["user_id"].map(user_id_map)

    # ── 3. สร้าง anime_info.csv + ดึงรูปจาก Jikan API ────────────────────
    print("\n3️⃣  สร้าง anime_info.csv + ดึงรูปจาก Jikan API...")

    # เอาเฉพาะ anime ที่มี rating จริง
    rated_anime_ids = set(rating_final["anime_id"].unique())
    top_anime_with_ratings = top_anime[top_anime["anime_id"].isin(rated_anime_ids)].copy()

    # ดึงรูปจาก Jikan API
    PLACEHOLDER = "https://cdn.myanimelist.net/img/sp/icon/apple-touch-icon-256.png"
    image_urls = {}
    anime_id_list = top_anime_with_ratings["anime_id"].tolist()
    total = len(anime_id_list)

    for i, aid in enumerate(anime_id_list):
        try:
            resp = requests.get(f"https://api.jikan.moe/v4/anime/{aid}", timeout=10)
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                img = data.get("images", {}).get("jpg", {}).get("image_url", PLACEHOLDER)
                image_urls[aid] = img
            elif resp.status_code == 429:
                # Rate limited — รอแล้วลองใหม่
                time.sleep(2)
                resp = requests.get(f"https://api.jikan.moe/v4/anime/{aid}", timeout=10)
                if resp.status_code == 200:
                    data = resp.json().get("data", {})
                    img = data.get("images", {}).get("jpg", {}).get("image_url", PLACEHOLDER)
                    image_urls[aid] = img
                else:
                    image_urls[aid] = PLACEHOLDER
            else:
                image_urls[aid] = PLACEHOLDER
        except Exception:
            image_urls[aid] = PLACEHOLDER

        # Progress
        pct = (i + 1) / total * 100
        print(f"\r   ดึงรูป: {i+1}/{total} ({pct:.0f}%)", end="", flush=True)
        time.sleep(0.4)  # Rate limit

    print()  # New line after progress

    anime_info = pd.DataFrame({
        "anime_id": top_anime_with_ratings["anime_id"].values,
        "title": top_anime_with_ratings["name"].values,
        "image_url": [image_urls.get(aid, PLACEHOLDER) for aid in top_anime_with_ratings["anime_id"]],
        "genres": top_anime_with_ratings["genre"].values,
        "score": top_anime_with_ratings["rating"].values,
    })
    anime_info.to_csv("anime_info.csv", index=False)
    print(f"   ✅ บันทึก {len(anime_info)} เรื่อง → anime_info.csv")

    # ── 4. สร้าง user_ratings.csv ───────────────────────────────────────
    print("\n4️⃣  สร้าง user_ratings.csv...")
    user_ratings = rating_final[["user_id", "anime_id", "rating"]].copy()
    user_ratings.to_csv("user_ratings.csv", index=False)
    print(f"   ✅ บันทึก {len(user_ratings):,} ratings → user_ratings.csv")

    # ── 5. สรุป ──────────────────────────────────────────────────────────
    n_users = user_ratings["user_id"].nunique()
    n_anime = user_ratings["anime_id"].nunique()
    avg_per_user = len(user_ratings) / max(n_users, 1)

    print(f"\n{'=' * 55}")
    print(f"  ✅ สำเร็จ! ข้อมูลจริงพร้อมใช้งาน")
    print(f"{'=' * 55}")
    print(f"  📦 อนิเมะ:      {n_anime} เรื่อง")
    print(f"  👤 ผู้ใช้จริง:     {n_users} คน")
    print(f"  ⭐ Ratings:      {len(user_ratings):,} รายการ")
    print(f"  📊 เฉลี่ย/คน:    {avg_per_user:.1f} ratings")
    print(f"\n  ข้อมูลพร้อม! รัน 'streamlit run app.py' ได้เลย 🚀")


def main():
    if not check_kaggle_files():
        return
    load_and_process()


if __name__ == "__main__":
    main()
