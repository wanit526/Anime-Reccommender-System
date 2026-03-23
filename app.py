"""
app.py
Streamlit UI for the Anime Recommender System.
Premium design with glassmorphism, animations, and polished aesthetics.
"""

import streamlit as st
import pandas as pd
import numpy as np
from recommender import load_data, build_item_similarity_matrix, get_recommendations
from nn_recommender import train_autoencoder, get_nn_recommendations

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme toggle ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    theme = st.radio("🎨 Theme", ["🌙 Dark", "☀️ Light"], index=0, horizontal=True)

is_dark = theme == "🌙 Dark"

# ── Theme tokens ─────────────────────────────────────────────────────────────
if is_dark:
    T = {
        "bg":           "#0b0e17",
        "surface":      "rgba(17, 22, 38, 0.85)",
        "glass":        "rgba(255,255,255,0.04)",
        "glass_border": "rgba(255,255,255,0.08)",
        "text":         "#eaedf3",
        "text2":        "#8b92a5",
        "accent1":      "#6c63ff",
        "accent2":      "#e94590",
        "accent_grad":  "linear-gradient(135deg, #6c63ff 0%, #e94590 100%)",
        "card_bg":      "rgba(20, 25, 45, 0.75)",
        "card_hover":   "0 20px 48px rgba(108, 99, 255, 0.22)",
        "badge_bg":     "rgba(108, 99, 255, 0.14)",
        "badge_text":   "#a8a1ff",
        "score_bg":     "rgba(255, 183, 77, 0.12)",
        "score_text":   "#ffb74d",
        "glow":         "rgba(108, 99, 255, 0.35)",
        "chart_color":  "#6c63ff",
    }
else:
    T = {
        "bg":           "#f4f6fb",
        "surface":      "rgba(255,255,255,0.90)",
        "glass":        "rgba(255,255,255,0.60)",
        "glass_border": "rgba(0,0,0,0.06)",
        "text":         "#1a1d2e",
        "text2":        "#6b7085",
        "accent1":      "#6c63ff",
        "accent2":      "#e94590",
        "accent_grad":  "linear-gradient(135deg, #6c63ff 0%, #e94590 100%)",
        "card_bg":      "rgba(255,255,255,0.80)",
        "card_hover":   "0 20px 48px rgba(108, 99, 255, 0.12)",
        "badge_bg":     "rgba(108, 99, 255, 0.08)",
        "badge_text":   "#5a52d5",
        "score_bg":     "rgba(255, 152, 0, 0.10)",
        "score_text":   "#e68a00",
        "glow":         "rgba(108, 99, 255, 0.15)",
        "chart_color":  "#6c63ff",
    }

# ── Inject Premium CSS ───────────────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* ── Reset & Base ─────────────────────── */
    html, body, [class*="st-"] {{
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }}
    .stApp {{
        background: {T["bg"]} !important;
    }}

    /* ── Animated Hero ────────────────────── */
    .hero {{
        text-align: center;
        padding: 3rem 1rem 1.5rem;
        position: relative;
    }}
    .hero-emoji {{
        font-size: 3.2rem;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
    }}
    @keyframes float {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-10px); }}
    }}
    .hero h1 {{
        background: {T["accent_grad"]};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0.3rem 0 0.1rem;
    }}
    .hero .subtitle {{
        color: {T["text2"]};
        font-size: 1rem;
        font-weight: 400;
        letter-spacing: 0.02em;
    }}

    /* ── Glass Divider ────────────────────── */
    .glass-divider {{
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, {T["accent1"]}44 50%, transparent 100%);
        margin: 1.2rem 0;
    }}

    /* ── Section Header ───────────────────── */
    .section-title {{
        color: {T["text"]};
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    .section-title .dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: {T["accent_grad"]};
        display: inline-block;
    }}

    /* ── Selected Anime Card ──────────────── */
    .selected-panel {{
        background: {T["card_bg"]};
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid {T["glass_border"]};
        border-radius: 20px;
        padding: 20px 24px;
        transition: box-shadow 0.35s ease;
    }}
    .selected-panel:hover {{
        box-shadow: 0 8px 32px {T["glow"]};
    }}
    .selected-panel h3 {{
        color: {T["text"]};
        margin: 0 0 10px;
        font-weight: 700;
        font-size: 1.35rem;
    }}
    .selected-panel .mal-badge {{
        background: {T["accent_grad"]};
        color: #fff;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 5px 14px;
        border-radius: 24px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        box-shadow: 0 4px 16px {T["glow"]};
    }}
    .genre-pill {{
        background: {T["badge_bg"]};
        color: {T["badge_text"]};
        font-size: 0.72rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
        display: inline-block;
        margin: 3px 2px;
        letter-spacing: 0.02em;
    }}

    /* ── Recommendation Cards ─────────────── */
    .rec-card {{
        background: {T["card_bg"]};
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid {T["glass_border"]};
        border-radius: 18px;
        padding: 10px;
        text-align: center;
        transition: transform 0.3s cubic-bezier(.34,1.56,.64,1), box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    .rec-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: {T["accent_grad"]};
        opacity: 0;
        transition: opacity 0.3s ease;
    }}
    .rec-card:hover {{
        transform: translateY(-8px) scale(1.01);
        box-shadow: {T["card_hover"]};
    }}
    .rec-card:hover::before {{
        opacity: 1;
    }}
    .rec-card img {{
        border-radius: 12px;
        width: 100%;
        aspect-ratio: 3/4.2;
        object-fit: cover;
        transition: transform 0.3s ease;
    }}
    .rec-card:hover img {{
        transform: scale(1.03);
    }}
    .rec-card .card-title {{
        color: {T["text"]};
        font-weight: 600;
        font-size: 0.78rem;
        margin-top: 10px;
        line-height: 1.35;
        min-height: 2.7em;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }}
    .rec-card .match-badge {{
        background: {T["accent_grad"]};
        color: #fff;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
        margin-top: 6px;
        box-shadow: 0 3px 12px {T["glow"]};
    }}
    .rec-card .card-genre {{
        background: {T["badge_bg"]};
        color: {T["badge_text"]};
        font-size: 0.6rem;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 10px;
        display: inline-block;
        margin: 2px 1px 0;
    }}
    .rec-card .card-score {{
        background: {T["score_bg"]};
        color: {T["score_text"]};
        font-weight: 700;
        font-size: 0.7rem;
        padding: 3px 9px;
        border-radius: 10px;
        display: inline-block;
        margin-top: 4px;
    }}

    /* ── Number Badge (rank) ──────────────── */
    .rank-num {{
        position: absolute;
        top: 16px;
        left: 16px;
        background: {T["accent_grad"]};
        color: #fff;
        font-weight: 800;
        font-size: 0.7rem;
        width: 26px;
        height: 26px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px {T["glow"]};
        z-index: 2;
    }}

    /* ── Sidebar beautify ─────────────────── */
    section[data-testid="stSidebar"] {{
        background: {T["surface"]} !important;
        backdrop-filter: blur(20px);
    }}

    /* ── Footer ───────────────────────────── */
    .footer {{
        text-align: center;
        color: {T["text2"]};
        font-size: 0.8rem;
        padding: 1rem 0;
    }}
    .footer a {{
        color: {T["accent1"]};
        text-decoration: none;
    }}
</style>
""", unsafe_allow_html=True)

# ── Hero Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-emoji">🎌</span>
    <h1>Anime Recommender</h1>
</div>
<hr class="glass-divider">
""", unsafe_allow_html=True)


# ── Load data (cached) ───────────────────────────────────────────────────────
@st.cache_data
def init_data():
    anime_df, ratings_df = load_data()
    sim_df = build_item_similarity_matrix(ratings_df)
    return anime_df, ratings_df, sim_df


@st.cache_resource
def init_nn_model(_ratings_df):
    """Train Autoencoder (cached so it only trains once)."""
    model, user_item, loss_history, anime_ids = train_autoencoder(_ratings_df, epochs=50)
    return model, user_item, loss_history, anime_ids


anime_df, ratings_df, similarity_df = init_data()
nn_model, nn_user_item, nn_loss_history, nn_anime_ids = init_nn_model(ratings_df)
titles = sorted(anime_df["title"].tolist())

# ── Search & Select ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title"><span class="dot"></span> เลือกอนิเมะที่คุณชอบ</div>', unsafe_allow_html=True)

search_query = st.text_input(
    "ค้นหาอนิเมะ",
    placeholder="🔍  พิมพ์ชื่อเพื่อค้นหา เช่น Naruto, One Piece, Attack on Titan...",
    label_visibility="collapsed",
)

if search_query:
    filtered_titles = [t for t in titles if search_query.lower() in t.lower()]
else:
    filtered_titles = titles

if not filtered_titles:
    st.warning(f'ไม่พบอนิเมะที่ตรงกับ "{search_query}" — ลองพิมพ์ใหม่')
    st.stop()

selected = st.selectbox("เลือกอนิเมะ:", filtered_titles, index=0, label_visibility="collapsed")

# ── Selected Anime Panel ─────────────────────────────────────────────────────
selected_info = anime_df[anime_df["title"] == selected].iloc[0]
genres_str = str(selected_info.get("genres", "N/A"))
if genres_str == "nan":
    genres_str = "N/A"
genre_pills = "".join(
    f'<span class="genre-pill">{g.strip()}</span>' for g in genres_str.split(",") if g.strip() != "N/A"
)
mal_score = selected_info.get("score", 0)

col_img, col_info = st.columns([1, 4], gap="medium")
with col_img:
    st.image(selected_info["image_url"], width=160)
with col_info:
    st.markdown(f"""
    <div class="selected-panel">
        <h3>{selected}</h3>
        <div style="margin-bottom:10px;">
            <span class="mal-badge">⭐ {mal_score}</span>
        </div>
        <div>{genre_pills if genre_pills else f'<span style="color:{T["text2"]}">ไม่มีข้อมูลแนว</span>'}</div>
        <p style="margin-top:10px; color:{T["text2"]}; font-size:0.8rem;">ID: {selected_info['anime_id']}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="glass-divider">', unsafe_allow_html=True)

# ── Recommendations ──────────────────────────────────────────────────────────
cosine_recs = get_recommendations(selected, anime_df, similarity_df)
nn_recs = get_nn_recommendations(selected, anime_df, nn_model, nn_user_item)

st.markdown('<div class="section-title"><span class="dot"></span> Top 5 แนะนำสำหรับคุณ</div>', unsafe_allow_html=True)

tab_cosine, tab_nn = st.tabs(["🔢 Cosine Similarity", "🧠 Neural Network (Autoencoder)"])


def render_rec_cards(recs: list[dict]):
    """Helper to render recommendation cards in 5 columns."""
    if not recs:
        st.info("ไม่พบคำแนะนำสำหรับเรื่องนี้ ลองเลือกอนิเมะเรื่องอื่น!")
        return

    cols = st.columns(5, gap="medium")
    for i, rec in enumerate(recs):
        rec_info = anime_df[anime_df["title"] == rec["title"]]
        rec_genres = ""
        rec_score = 0
        if not rec_info.empty:
            rec_genres = rec_info.iloc[0].get("genres", "N/A")
            rec_score = rec_info.iloc[0].get("score", 0)

        genre_badges = "".join(
            f'<span class="card-genre">{g.strip()}</span>'
            for g in str(rec_genres).split(",")[:3]
            if g.strip() != "N/A"
        )

        with cols[i]:
            st.markdown(f"""
            <div class="rec-card">
                <div class="rank-num">{i + 1}</div>
                <img src="{rec['image_url']}" alt="{rec['title']}">
                <div class="card-title">{rec['title']}</div>
                <span class="match-badge">{rec['match_percentage']}% match</span>
                <div style="margin-top:5px;">{genre_badges}</div>
                <span class="card-score">⭐ {rec_score}</span>
            </div>
            """, unsafe_allow_html=True)


# ── Tab 1: Cosine Similarity ─────────────────────────────────────────────
with tab_cosine:
    st.markdown(f"""
    <div style="background:{T['glass']}; border:1px solid {T['glass_border']}; border-radius:12px;
                padding:12px 18px; margin-bottom:16px; font-size:0.82rem; color:{T['text2']};">
        💡 <b>Cosine Similarity</b> — คำนวณ "มุม" ระหว่าง Vector คะแนนของอนิเมะแต่ละคู่
        จาก User-Item Matrix เพื่อหาเรื่องที่คนกลุ่มเดียวกันให้คะแนนคล้ายกัน
    </div>
    """, unsafe_allow_html=True)
    render_rec_cards(cosine_recs)

    if cosine_recs:
        st.markdown('<hr class="glass-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><span class="dot"></span> Similarity Visualization</div>', unsafe_allow_html=True)
        chart_data = pd.DataFrame({
            "อนิเมะ": [r["title"][:35] + ("..." if len(r["title"]) > 35 else "") for r in cosine_recs],
            "Similarity (%)": [r["match_percentage"] for r in cosine_recs],
        })
        st.bar_chart(chart_data.set_index("อนิเมะ"), horizontal=True, color=T["chart_color"])


# ── Tab 2: Neural Network ────────────────────────────────────────────────
with tab_nn:
    st.markdown(f"""
    <div style="background:{T['glass']}; border:1px solid {T['glass_border']}; border-radius:12px;
                padding:12px 18px; margin-bottom:16px; font-size:0.82rem; color:{T['text2']};">
        🧠 <b>Neural Network (Autoencoder)</b> — ฝึก AI ให้เรียนรู้ Pattern ของ User จาก Rating Data
        แล้วทำนายว่า User น่าจะให้คะแนนอนิเมะเรื่องไหนสูง<br>
        Architecture: <code>Input({len(nn_anime_ids)}) → 128 → 32 (Latent) → 128 → Output({len(nn_anime_ids)})</code>
    </div>
    """, unsafe_allow_html=True)
    render_rec_cards(nn_recs)

    if nn_recs:
        st.markdown('<hr class="glass-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><span class="dot"></span> Similarity Visualization</div>', unsafe_allow_html=True)
        nn_chart = pd.DataFrame({
            "อนิเมะ": [r["title"][:35] + ("..." if len(r["title"]) > 35 else "") for r in nn_recs],
            "Match (%)": [r["match_percentage"] for r in nn_recs],
        })
        st.bar_chart(nn_chart.set_index("อนิเมะ"), horizontal=True, color="#e94590")

    # ── Training Loss Chart ──────────────────────────────────────────
    st.markdown('<hr class="glass-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="dot"></span> Training Loss (Autoencoder)</div>', unsafe_allow_html=True)
    loss_chart = pd.DataFrame({
        "Epoch": list(range(1, len(nn_loss_history) + 1)),
        "Loss (MSE)": nn_loss_history,
    })
    st.line_chart(loss_chart.set_index("Epoch"), color="#e94590")

# ── Sidebar stats ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📈 สถิติ")
    st.metric("จำนวนอนิเมะ", f"{len(anime_df)} เรื่อง")
    st.metric("ผู้ใช้จำลอง", "500 คน")

    st.markdown("### 🧠 Neural Network")
    st.markdown(f"""
    <div style="background:{T['glass']}; border:1px solid {T['glass_border']}; border-radius:10px;
                padding:10px 14px; font-size:0.78rem; color:{T['text2']};">
        <b style="color:{T['text']};">Architecture</b><br>
        Input({len(nn_anime_ids)}) → 128 → 32 → 128 → Output({len(nn_anime_ids)})<br><br>
        <b style="color:{T['text']};">Final Loss (MSE)</b><br>
        {nn_loss_history[-1]:.6f}<br><br>
        <b style="color:{T['text']};">Epochs</b>: 50 &nbsp;|&nbsp;
        <b style="color:{T['text']};">Optimizer</b>: Adam
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏷️ แนวยอดนิยม")
    all_genres = []
    for g in anime_df["genres"].dropna():
        all_genres.extend([x.strip() for x in str(g).split(",") if x.strip() not in ("N/A", "nan")])
    if all_genres:
        genre_counts = pd.Series(all_genres).value_counts().head(10)
        st.bar_chart(genre_counts, color=T["chart_color"])

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<hr class="glass-divider">
<div class="footer">
    สร้างด้วย ❤️ โดยใช้ <a href="https://streamlit.io">Streamlit</a>,
    <a href="https://scikit-learn.org">scikit-learn</a> &
    <a href="https://pytorch.org">PyTorch</a> ·
    Item-based Collaborative Filtering + Neural Network Autoencoder
</div>
""", unsafe_allow_html=True)
