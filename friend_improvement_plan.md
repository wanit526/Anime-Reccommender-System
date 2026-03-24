# 🎌 แผนปรับปรุงโปรเจค Anime Recommender — ให้ตรงกับที่เรียน

> **เป้าหมาย:** แก้ไข 5 จุด ให้ตรงกับที่เรียนมาจริงๆ โดยคงส่วน UI ที่ดีอยู่แล้วไว้

---

## 🔴 สิ่งที่ต้องแก้ (5 ข้อ)

| # | ปัญหา | วิธีแก้ | ระดับ |
|---|-------|--------|------|
| 1 | ข้อมูล Synthetic (สุ่มขึ้นมา) | ใช้ Dataset จริงจาก Kaggle | ⭐⭐ |
| 2 | match% ปลอม (บังคับ 70-100%) | แสดงค่า predicted rating จริง | ⭐ |
| 3 | ใช้ PyTorch (ไม่ตรงที่เรียน) | เปลี่ยนเป็น TensorFlow/Keras | ⭐⭐⭐ |
| 4 | ไม่มี Train/Test Split | เพิ่ม 80/20 split | ⭐ |
| 5 | ไม่มี Evaluation Metrics | เพิ่ม RMSE, MAE, R² | ⭐⭐ |

---

## 📁 โครงสร้างไฟล์ — เดิม vs ใหม่

```diff
 Anime-Reccommender-System-main/
 ├── app.py                  # ✅ เก็บไว้ (แก้นิดหน่อย)
 ├── recommender.py          # ✅ เก็บไว้ (เหมือนเดิม)
-├── nn_recommender.py       # ❌ ลบ (PyTorch)
+├── nn_recommender.py       # 🔄 เขียนใหม่ (TensorFlow/Keras)
-├── generate_data.py        # ❌ ลบ (สร้างข้อมูลปลอม)
+├── load_data.py            # 🆕 โหลด dataset จริงจาก Kaggle
 ├── anime_info.csv          # ✅ เก็บไว้
-├── user_ratings.csv        # ❌ ลบ (ข้อมูลปลอม)
+├── user_ratings.csv        # 🔄 จาก Kaggle dataset จริง
 ├── requirements.txt        # 🔄 แก้ (เปลี่ยน PyTorch → TensorFlow)
 └── README.md               # 🔄 อัปเดต
```

---

## 📊 Dataset: ใช้ของจริงจาก Kaggle

**แนะนำ:** [MyAnimeList Dataset](https://www.kaggle.com/datasets/azathoth42/myanimelist) (300,000 users, 20 ล้าน ratings)

> [!IMPORTANT]
> Dataset ใหญ่มาก → ตัดเหลือ **Top 1,000 อนิเมะ + 5,000 users** ให้เทรนได้เร็ว

### ไฟล์ใหม่: `load_data.py`
```python
# แทน generate_data.py (ที่สุ่มข้อมูล)
# โหลดจาก Kaggle CSV จริง → ตัดเหลือขนาดพอดี → บันทึก
```

---

## 🔧 วิเคราะห์การแก้ไขแต่ละไฟล์

### 1. ❌ ลบ [generate_data.py](file:///c:/Users/YEDHEE/Desktop/machine%20learning/Mini%20Project%20my%20friend/Anime-Reccommender-System-main/generate_data.py)

ไฟล์นี้สร้างข้อมูลปลอม → ไม่ต้องแล้ว เพราะใช้ dataset จริง

---

### 2. 🆕 สร้าง `load_data.py` (แทน generate_data.py)

```python
# หน้าที่:
# 1. โหลด Kaggle CSV (anime.csv + ratings.csv)
# 2. ตัดเหลือ Top 1,000 anime + 5,000 users ที่ให้คะแนนมากที่สุด
# 3. สร้าง anime_info.csv + user_ratings.csv (ข้อมูลจริง!)
```

---

### 3. 🔄 เขียนใหม่ [nn_recommender.py](file:///c:/Users/YEDHEE/Desktop/machine%20learning/Mini%20Project%20my%20friend/Anime-Reccommender-System-main/nn_recommender.py) (PyTorch → TensorFlow/Keras)

**เดิม (PyTorch):**
```python
class AnimeAutoencoder(nn.Module):         # PyTorch ❌
    self.encoder = nn.Sequential(...)
    self.decoder = nn.Sequential(...)
    loss.backward()                        # PyTorch Backprop
    optimizer.step()
```

**ใหม่ (TensorFlow/Keras):**
```python
def build_autoencoder(input_dim):          # Keras ✅
    inp = Input(shape=(input_dim,))
    # Encoder
    x = Dense(128, activation='relu')(inp)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)    # Latent Space
    # Decoder
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(input_dim, activation='sigmoid')(x)
    
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    return model

# เพิ่ม! ✅
def train_with_split(ratings_df):
    # Train/Test Split 80/20
    train_data, test_data = train_test_split(matrix, test_size=0.2)
    
    model.fit(train_data, train_data,
              validation_data=(test_data, test_data),
              epochs=50, batch_size=32)
    
    # Evaluation Metrics ✅
    predictions = model.predict(test_data)
    rmse = sqrt(mean_squared_error(test_data, predictions))
    mae = mean_absolute_error(test_data, predictions)
```

**สิ่งที่เพิ่มใหม่:**

| เพิ่มอะไร | เดิม | ใหม่ |
|----------|------|------|
| Framework | PyTorch | **TensorFlow/Keras** ✅ |
| Train/Test Split | ❌ ไม่มี | **80/20 Split** ✅ |
| Validation | ❌ ไม่มี | **validation_data** ✅ |
| RMSE | ❌ ไม่มี | **มี** ✅ |
| MAE | ❌ ไม่มี | **มี** ✅ |
| R² Score | ❌ ไม่มี | **มี** ✅ |

---

### 4. 🔄 แก้ [app.py](file:///c:/Users/YEDHEE/Desktop/machine%20learning/WasteClassifier/app.py) (เล็กน้อย)

| แก้ตรงไหน | เดิม | ใหม่ |
|----------|------|------|
| Import | `import torch` | `import tensorflow` |
| match% (NN tab) | บังคับ 70-100% | แสดง predicted rating จริง (1-10) |
| เพิ่ม Tab | — | **Tab "📊 Model Evaluation"** แสดง RMSE, MAE, R² |
| เพิ่ม Section | — | **Prediction vs Actual Chart** |

---

### 5. 🔄 แก้ [requirements.txt](file:///c:/Users/YEDHEE/Desktop/machine%20learning/WasteClassifier/requirements.txt)

```diff
-torch
-torchvision
+tensorflow
 pandas
 numpy
 scikit-learn
 streamlit
 requests
```

---

### 6. ✅ เก็บไว้ [recommender.py](file:///c:/Users/YEDHEE/Desktop/machine%20learning/Mini%20Project%20my%20friend/Anime-Reccommender-System-main/recommender.py) (ไม่ต้องแก้)

Cosine Similarity ใช้ได้ดีอยู่แล้ว ใช้เป็น baseline เปรียบเทียบกับ Autoencoder

---

## 📋 Step-by-Step Workflow

| Step | ทำอะไร | ไฟล์ |
|------|--------|------|
| **1** | ดาวน์โหลด [MAL Dataset](https://www.kaggle.com/datasets/azathoth42/myanimelist) จาก Kaggle | — |
| **2** | เขียน `load_data.py` — โหลด + ตัดข้อมูล | `load_data.py` (ใหม่) |
| **3** | ลบ [generate_data.py](file:///c:/Users/YEDHEE/Desktop/machine%20learning/Mini%20Project%20my%20friend/Anime-Reccommender-System-main/generate_data.py) | ลบ |
| **4** | เขียน [nn_recommender.py](file:///c:/Users/YEDHEE/Desktop/machine%20learning/Mini%20Project%20my%20friend/Anime-Reccommender-System-main/nn_recommender.py) ใหม่ด้วย TF/Keras + Train/Test + Evaluation | [nn_recommender.py](file:///c:/Users/YEDHEE/Desktop/machine%20learning/Mini%20Project%20my%20friend/Anime-Reccommender-System-main/nn_recommender.py) (เขียนใหม่) |
| **5** | แก้ [app.py](file:///c:/Users/YEDHEE/Desktop/machine%20learning/WasteClassifier/app.py) — import, match%, เพิ่ม Eval tab | [app.py](file:///c:/Users/YEDHEE/Desktop/machine%20learning/WasteClassifier/app.py) |
| **6** | แก้ [requirements.txt](file:///c:/Users/YEDHEE/Desktop/machine%20learning/WasteClassifier/requirements.txt) | [requirements.txt](file:///c:/Users/YEDHEE/Desktop/machine%20learning/WasteClassifier/requirements.txt) |
| **7** | ทดสอบรัน `streamlit run app.py` | ทดสอบ |

---

## 🆚 ผลลัพธ์ก่อน vs หลังแก้

| หัวข้อ | ก่อนแก้ | หลังแก้ |
|--------|--------|--------|
| ข้อมูล | ❌ Synthetic (สุ่ม) | ✅ **Kaggle จริง** |
| Framework | ❌ PyTorch | ✅ **TensorFlow/Keras** |
| Train/Test Split | ❌ ไม่มี | ✅ **80/20** |
| Evaluation | ❌ ไม่มี | ✅ **RMSE, MAE, R²** |
| match% | ❌ ปั้น 70-100% | ✅ **ค่าจริง** |
| ตรงที่เรียน | ⚠️ บางส่วน | ✅ **ตรงมาก** (NN module) |
| UI | ✅ สวย | ✅ สวย + **เพิ่ม Eval tab** |

---

## Verification Plan

1. **รัน `load_data.py`** → ตรวจว่า dataset จริงถูกโหลดและตัดขนาดถูกต้อง
2. **รัน training** → ตรวจว่า model เทรนด้วย TF/Keras สำเร็จ มี train/val loss ลดลง
3. **ตรวจ Evaluation** → RMSE, MAE, R² แสดงค่าที่สมเหตุสมผล
4. **รัน `streamlit run app.py`** → ตรวจ UI, recommendations, Eval tab
