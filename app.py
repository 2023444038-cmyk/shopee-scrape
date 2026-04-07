import os
import re
import pickle
import gdown
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, flash, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from shopee_scrapper import scrape_shopee_bulk

# ─── PATH SETUP ───────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'absa_kipas_model.h5'
TOKENIZER_PATH = BASE_DIR / 'tokenizer.pickle'

# ─── GOOGLE DRIVE AUTO DOWNLOAD ───────────────
MODEL_GDRIVE_ID = os.environ.get("MODEL_GDRIVE_ID", "")
TOKENIZER_GDRIVE_ID = os.environ.get("TOKENIZER_GDRIVE_ID", "")

if not MODEL_PATH.exists() and MODEL_GDRIVE_ID:
    print("⬇️ Downloading model...")
    gdown.download(id=MODEL_GDRIVE_ID, output=str(MODEL_PATH), quiet=False)

if not TOKENIZER_PATH.exists() and TOKENIZER_GDRIVE_ID:
    print("⬇️ Downloading tokenizer...")
    gdown.download(id=TOKENIZER_GDRIVE_ID, output=str(TOKENIZER_PATH), quiet=False)

# ─── LOAD MODEL ───────────────────────────────
print("🚀 Loading model...")

def custom_input_layer(config):
    config.pop('batch_shape', None)
    config.pop('optional', None)
    return tf.keras.layers.InputLayer(**config)

def custom_attention_loader(config):
    if 'score_mode' in config and not isinstance(config['score_mode'], str):
        config['score_mode'] = 'dot'
    return tf.keras.layers.Attention(**config)

try:
    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        custom_objects={
            'InputLayer': custom_input_layer,
            'Attention': custom_attention_loader
        },
        compile=False
    )
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Primary load failed: {e}")
    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
        safe_mode=False
    )

# ─── LOAD TOKENIZER ───────────────────────────
with open(str(TOKENIZER_PATH), 'rb') as f:
    tokenizer = pickle.load(f)

# ─── FLASK APP ───────────────────────────────
app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get("SECRET_KEY", "absa_secret")

# ─── CONFIG ──────────────────────────────────
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

# ─── CLEAN TEXT ──────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [malay_slang.get(w, w) for w in words]
    return " ".join(words)

# ─── PREDICTION FUNCTION ─────────────────────
def run_prediction(scraped_data):
    valid_data = [item for item in scraped_data if str(item.get('review', '')).strip()]
    if not valid_data:
        return None, "Tiada ulasan dijumpai."

    reviews = [item['review'] for item in valid_data]
    total_reviews = len(reviews)

    stars = [
        int(item['star'][0]) if isinstance(item.get('star'), list)
        else int(item.get('star', 5))
        for item in valid_data
    ]

    avg_star = round(sum(stars) / total_reviews, 1)

    # Clean + tokenize
    cleaned = [clean_text(r) for r in reviews]
    seq = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(seq, maxlen=60, padding="post")

    predictions = model.predict(padded)

    aspect_count = {a: 0 for a in ASPECT_COLUMNS}
    for pred in predictions:
        for i, aspect in enumerate(ASPECT_COLUMNS):
            if pred[i] > 0.4:
                aspect_count[aspect] += 1

    return {
        "total": total_reviews,
        "avg_star": avg_star,
        "aspect_count": aspect_count
    }, None

# ─── ROUTES ──────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_url", methods=["POST"])
def predict_url():
    url = request.form.get("shopee_url", "").strip()

    if not url:
        flash("Masukkan URL Shopee")
        return render_template("index.html")

    scraped_data = scrape_shopee_bulk(url, total_wanted=100)

    if not scraped_data:
        flash("Scrape gagal (IP blocked)")
        return render_template("index.html")

    result, error = run_prediction(scraped_data)

    if error:
        flash(error)
        return render_template("index.html")

    return render_template("index.html", **result)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# ─── RUN APP ─────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)