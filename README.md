# FanSight — Shopee ABSA Analyzer

Sistem analisis sentimen berasaskan aspek (ABSA) untuk ulasan produk kipas di Shopee.
Dibina dengan Flask + BiGRU + TensorFlow.

---

## Cara Deploy ke Render.com (Percuma)

### Langkah 1 — Upload model ke Google Drive

Model `.h5` dan `tokenizer.pickle` terlalu besar untuk GitHub (>100MB).
Upload kedua-dua fail ke Google Drive:

1. Buka [drive.google.com](https://drive.google.com)
2. Upload `absa_kipas_model.h5` dan `tokenizer.pickle`
3. Klik kanan setiap fail → **Share** → **Anyone with the link**
4. Salin **File ID** dari URL:
   ```
   https://drive.google.com/file/d/XXXXXXXXXXXXXXX/view
                                   ↑ ini File ID
   ```

### Langkah 2 — Push ke GitHub

```bash
git init
git add .
git commit -m "initial commit"
git branch -M main
git remote add origin https://github.com/USERNAME/shopee-absa.git
git push -u origin main
```

### Langkah 3 — Deploy di Render

1. Pergi ke [render.com](https://render.com) → daftar percuma
2. Klik **New +** → **Web Service**
3. Connect GitHub → pilih repo `shopee-absa`
4. Setting akan auto-detect dari `render.yaml`
5. Tambah **Environment Variables**:
   - `MODEL_GDRIVE_ID` → File ID model `.h5`
   - `TOKENIZER_GDRIVE_ID` → File ID `tokenizer.pickle`
6. Klik **Create Web Service**

Tunggu 5-10 minit → dapat URL seperti `https://shopee-absa.onrender.com`

---

## Cara Run Secara Lokal

```bash
# Install dependencies
pip install -r requirements.txt

# Pastikan model dan tokenizer ada dalam folder
# absa_kipas_model.h5
# tokenizer.pickle

# Run Flask
python app.py
```

Buka browser: `http://localhost:5000`

---

## API Endpoint

### POST /api/predict
Untuk integrasi dengan frontend atau aplikasi lain.

```bash
curl -X POST https://shopee-absa.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{"shopee_url": "https://shopee.com.my/...", "review_limit": 100}'
```

Response JSON:
```json
{
  "total": 95,
  "avg_star": 4.2,
  "positive_pct": 78,
  "aspect_percentages": {
    "Kualiti_Fizikal": 82,
    "Prestasi_Angin": 75,
    ...
  },
  "ai_summary": "Majoriti pelanggan...",
  "top_reviews": [...]
}
```

### GET /health
Semak status server.

---

## 7 Aspek Yang Dianalisis

| Aspek | Keterangan |
|-------|-----------|
| Kualiti_Fizikal | Material, ketahanan, reka bentuk |
| Prestasi_Angin | Kekuatan angin, tahap bunyi |
| Bateri_Pengecasan | Jangka hayat bateri, kelajuan cas |
| Harga | Nilai untuk wang |
| Penghantaran | Kelajuan dan kebolehpercayaan penghantaran |
| Pembungkusan | Kualiti pembungkusan |
| Layanan_Penjual | Responsif dan perkhidmatan penjual |

---

## Struktur Projek

```
shopee-absa/
├── app.py                  # Flask app utama
├── shopee_scrapper.py      # Shopee API scraper
├── requirements.txt        # Python dependencies
├── Procfile                # Gunicorn start command
├── render.yaml             # Render.com config
├── templates/
│   └── index.html          # Frontend UI
└── README.md
```
