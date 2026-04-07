import os
import re
import pickle
import gdown
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, flash, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from shopee_scrapper import scrape_shopee_bulk

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'absa_kipas_model.h5'
TOKENIZER_PATH = BASE_DIR / 'tokenizer.pickle'

# ─── Auto-download model from Google Drive if not found ───────────────────────
# Upload absa_kipas_model.h5 ke Google Drive, share publicly, letak ID di sini
MODEL_GDRIVE_ID = os.environ.get("MODEL_GDRIVE_ID", "")
TOKENIZER_GDRIVE_ID = os.environ.get("TOKENIZER_GDRIVE_ID", "")

if not MODEL_PATH.exists() and MODEL_GDRIVE_ID:
    print("Model tidak dijumpai. Memuat turun dari Google Drive...")
    gdown.download(id=MODEL_GDRIVE_ID, output=str(MODEL_PATH), quiet=False)

if not TOKENIZER_PATH.exists() and TOKENIZER_GDRIVE_ID:
    print("Tokenizer tidak dijumpai. Memuat turun dari Google Drive...")
    gdown.download(id=TOKENIZER_GDRIVE_ID, output=str(TOKENIZER_PATH), quiet=False)

# ─── Load model & tokenizer ───────────────────────────────────────────────────
print("Memuatkan model dan tokenizer...")
model = load_model(str(MODEL_PATH))
with open(str(TOKENIZER_PATH), 'rb') as f:
    tokenizer = pickle.load(f)

app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get("SECRET_KEY", "absa_secret_2024")

ASPECT_COLUMNS = [
    'Kualiti_Fizikal', 'Prestasi_Angin', 'Bateri_Pengecasan',
    'Harga', 'Penghantaran', 'Pembungkusan', 'Layanan_Penjual'
]

ASPECT_LABELS = {
    'Kualiti_Fizikal': 'Kualiti Fizikal',
    'Prestasi_Angin': 'Prestasi Angin',
    'Bateri_Pengecasan': 'Bateri & Pengecasan',
    'Harga': 'Harga & Nilai',
    'Penghantaran': 'Penghantaran',
    'Pembungkusan': 'Pembungkusan',
    'Layanan_Penjual': 'Layanan Penjual'
}

malay_slang = {
    "x": "tidak", "tak": "tidak", "tk": "tidak",
    "ok": "elok", "okey": "elok", "laju": "cepat"
}


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [malay_slang.get(w, w) for w in words]
    return " ".join(words)


def run_prediction(scraped_data):
    """Core ABSA prediction logic — shared by HTML and JSON routes."""
    valid_data = [item for item in scraped_data if str(item.get('review', '')).strip()]
    if not valid_data:
        return None, "Tiada ulasan bertulis dijumpai."

    reviews = [item['review'] for item in valid_data]
    total_reviews = len(reviews)

    stars = [
        int(item['star'][0]) if isinstance(item.get('star'), list)
        else int(item.get('star', 5))
        for item in valid_data
    ]
    avg_star = round(sum(stars) / total_reviews, 1)

    star_counts = {5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
    for s in stars:
        if s in star_counts:
            star_counts[s] += 1
    star_percentages = {k: round((v / total_reviews) * 100) for k, v in star_counts.items()}

    cleaned = [clean_text(r) for r in reviews]
    seq = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(seq, maxlen=60, padding="post")

    predictions = model.predict(padded)
    aspect_count = {a: 0 for a in ASPECT_COLUMNS}
    for pred in predictions:
        for i, aspect in enumerate(ASPECT_COLUMNS):
            if pred[i] > 0.4:
                aspect_count[aspect] += 1

    base_satisfaction = (avg_star / 5.0) * 100
    max_mention = max(aspect_count.values()) if any(aspect_count.values()) else 1

    aspect_percentages = {}
    for aspect, count in aspect_count.items():
        if count == 0:
            aspect_percentages[aspect] = 0
        else:
            ratio = count / max_mention
            score = base_satisfaction - (20 * (1 - ratio))
            aspect_percentages[aspect] = round(min(100, score))

    sorted_aspects = sorted(aspect_percentages.items(), key=lambda x: x[1], reverse=True)
    top_1 = sorted_aspects[0][0].replace('_', ' ')
    top_2 = sorted_aspects[1][0].replace('_', ' ') if len(sorted_aspects) > 1 else ""
    bottom_key, bottom_score = sorted_aspects[-1]

    summary = f"Majoriti pelanggan mendapati produk ini memuaskan terutamanya dari segi {top_1}"
    if top_2:
        summary += f" dan {top_2}."
    else:
        summary += "."
    summary += " Secara keseluruhan, pembeli memuji kualiti yang ditawarkan."
    if 0 < bottom_score < 75:
        summary += f" Walau bagaimanapun, perhatian perlu diberikan terhadap aspek {bottom_key.replace('_', ' ')}."

    return {
        "total": total_reviews,
        "avg_star": avg_star,
        "star_percentages": star_percentages,
        "aspect_percentages": aspect_percentages,
        "aspect_labels": ASPECT_LABELS,
        "ai_summary": summary,
        "top_reviews": valid_data[:5],
        "positive_pct": round((len([s for s in stars if s >= 4]) / total_reviews) * 100)
    }, None


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict_url", methods=["POST"])
def predict_url():
    """HTML form route — returns rendered template."""
    url = request.form.get("shopee_url", "").strip()
    limit_val = request.form.get("review_limit", "100")
    total_wanted = int(limit_val)

    if not url:
        flash("Sila masukkan link Shopee yang sah.")
        return render_template("index.html")

    print(f"Scraping {total_wanted} ulasan dari: {url}")
    scraped_data = scrape_shopee_bulk(url, total_wanted=total_wanted)

    if not scraped_data:
        flash("❌ Gagal dapatkan review. Shopee mungkin block IP. Cuba lagi.")
        return render_template("index.html")

    result, error = run_prediction(scraped_data)
    if error:
        flash(f"❌ {error}")
        return render_template("index.html")

    return render_template("index.html", url=url, limit_selected=limit_val, **result)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API route — untuk frontend / widget eksternal."""
    data = request.get_json()
    url = (data or {}).get("shopee_url", "").strip()
    total_wanted = int((data or {}).get("review_limit", 100))

    if not url:
        return jsonify({"error": "shopee_url diperlukan"}), 400

    scraped_data = scrape_shopee_bulk(url, total_wanted=total_wanted)
    if not scraped_data:
        return jsonify({"error": "Gagal scrape. IP mungkin diblock."}), 503

    result, error = run_prediction(scraped_data)
    if error:
        return jsonify({"error": error}), 422

    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
