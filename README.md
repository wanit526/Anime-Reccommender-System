# 🎌 ระบบแนะนำอนิเมะ (Anime Recommender System)

ระบบแนะนำอนิเมะโดยใช้ **Item-based Collaborative Filtering** ร่วมกับ **Cosine Similarity** พัฒนาด้วย Python, scikit-learn และ Streamlit

## ✨ คุณสมบัติหลัก

- **อนิเมะ 500+ เรื่อง** ดึงข้อมูลจาก [Jikan API](https://jikan.moe/) (MyAnimeList)
- **ผู้ใช้จำลอง 300 คน** พร้อมการให้คะแนนที่สมจริง
- **Item-based Collaborative Filtering** ใช้ Cosine Similarity ในการคำนวณความคล้ายคลึง
- 🔍 **ช่องค้นหา** สำหรับค้นหาอนิเมะได้ง่าย
- 🏷️ **แสดงแนว (Genre)** และ **คะแนน MAL** บนทุกการ์ด
- 🌙 **Dark/Light Mode** สลับ theme ได้จาก sidebar
- 📊 **Similarity Chart** กราฟแท่งแสดงความคล้ายคลึง
- 📈 **สถิติ** จำนวนอนิเมะ/ผู้ใช้ + กราฟแนวยอดนิยม

## 📁 โครงสร้างโปรเจกต์

```
ML_MiniProject/
├── app.py              # แอปพลิเคชัน Streamlit
├── recommender.py      # เครื่องมือแนะนำ (cosine similarity)
├── generate_data.py    # ดึงข้อมูลและสร้างคะแนนจำลอง
├── anime_info.csv      # ข้อมูลอนิเมะ (id, ชื่อ, URL รูปภาพ, แนว, คะแนน)
├── user_ratings.csv    # คะแนนจากผู้ใช้จำลอง
├── requirements.txt    # ไลบรารีที่ต้องใช้
└── README.md
```

## 🚀 วิธีเริ่มต้นใช้งาน

### 1. ติดตั้งไลบรารีที่จำเป็น

```bash
pip install -r requirements.txt
```

### 2. สร้างข้อมูล

ดึงข้อมูลอนิเมะจาก Jikan API และสร้างคะแนนจำลองจากผู้ใช้:

```bash
python generate_data.py
```

> **หมายเหตุ:** สคริปต์นี้มีการเรียก API พร้อมจำกัดอัตราการเรียก (`time.sleep(1)`) ใช้เวลาประมาณ 20 วินาที

### 3. รันแอปพลิเคชัน

```bash
streamlit run app.py
```

แอปจะเปิดที่ [http://localhost:8501](http://localhost:8501)

## 🔧 หลักการทำงาน

1. **การเก็บข้อมูล** — `generate_data.py` ดึงข้อมูลอนิเมะยอดนิยมจาก Jikan API (500 เรื่อง) พร้อมแนว (Genre) และคะแนน MAL
2. **User-Item Matrix** — สร้างตาราง Pivot Table ที่แมประหว่างผู้ใช้กับคะแนนอนิเมะ
3. **Cosine Similarity** — คำนวณความคล้ายคลึงระหว่างอนิเมะแต่ละคู่ โดยพิจารณาจากรูปแบบการให้คะแนนของผู้ใช้
4. **การแนะนำ** — เมื่อเลือกอนิเมะ ระบบจะแสดง 5 เรื่องที่มีความคล้ายคลึงมากที่สุด พร้อมเปอร์เซ็นต์ความตรงกัน

## 🛠️ เทคโนโลยีที่ใช้

| ส่วนประกอบ | เทคโนโลยี |
|-----------|-----------|
| ภาษา | Python 3.11+ |
| Machine Learning | scikit-learn (Cosine Similarity) |
| จัดการข้อมูล | pandas, NumPy |
| หน้า UI | Streamlit |
| API | Jikan v4 (MyAnimeList) |
