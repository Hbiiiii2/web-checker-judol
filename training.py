## PREDICT.PY — Malicious URL / Judol Detector
## Compatible dengan model yang ditraining dari train.py

import re
import math
import argparse
import joblib
import numpy as np
import requests

from bs4 import BeautifulSoup
from scipy.sparse import hstack
from urllib.parse import urlparse

# ============================================
# CONFIG — sesuaikan dengan output train.py
# ============================================

MODEL_PATH  = "model.pkl"
TFIDF_PATH  = "tfidf.pkl"
SCALER_PATH = "scaler.pkl"

# Harus sama persis dengan urutan di train.py
# ⚠️ Kalau kamu retrain tanpa judol_score/safe_score (disarankan),
#    hapus baris yang bocor dari sini juga.
NUMERICAL_COLUMNS = [
    "risk_score",
    "judol_score",
    "phishing_score",
    "safe_score",
    "metadata_quality",
    "url_length",
    "domain_length",
    "path_length",
    "dot_count",
    "dash_count",
    "digit_count",
    "special_char_count",
    "https",
    "keyword_density",
    "repeated_words_count",
    "symbol_ratio",
    "domain_entropy",
    "suspicious_subdomain_score",
    "redirect_indicator_score",
]

# ── Kalau pakai model BARU (tanpa fitur bocor), ganti ke ini:
# NUMERICAL_COLUMNS = [
#     "url_length", "domain_length", "path_length",
#     "dot_count", "dash_count", "digit_count",
#     "special_char_count", "https", "keyword_density",
#     "repeated_words_count", "symbol_ratio", "domain_entropy",
#     "suspicious_subdomain_score", "redirect_indicator_score",
# ]

# ============================================
# JUDOL KEYWORD LIST
# ============================================

JUDOL_KEYWORDS = [
    "slot", "gacor", "maxwin", "jackpot", "togel", "toto",
    "poker", "casino", "betting", "taruhan", "judi", "rtp",
    "scatter", "pragmatic", "pg soft", "zeus", "mahjong",
    "olympus", "bonus", "deposit", "withdraw", "spin",
    "live casino", "sportsbook", "parlay", "mix parlay",
    "agen slot", "situs slot", "daftar slot", "login slot",
    "bocoran", "gampang menang", "jp", "hoki", "lucky",
    "77", "88", "99", "168", "777", "888",
]

SUSPICIOUS_SUBDOMAINS = [
    "slot", "gacor", "judol", "maxwin", "togel", "bet",
    "casino", "poker", "live", "jackpot", "bonus",
]

REDIRECT_INDICATORS = [
    "redirect", "redir", "go", "out", "click", "track",
    "ref", "aff", "affiliate", "link", "url", "visit",
]

# ============================================
# AUTO SCRAPER — fetch title & meta dari URL
# ============================================

SCRAPER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8",
}

def auto_scrape(url: str, timeout: int = 8) -> dict:
    """
    Otomatis fetch title, meta description, dan keywords dari URL.
    Return dict: { title, meta_description, triggered_signals }
    Kalau gagal (timeout, block, dll), return string kosong.
    """
    result = {
        "title":             "",
        "meta_description":  "",
        "triggered_signals": "",
    }

    try:
        print(f"  [SCRAPE] Fetching: {url}")
        resp = requests.get(
            url,
            headers=SCRAPER_HEADERS,
            timeout=timeout,
            allow_redirects=True,
        )
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # --- Title ---
        title_tag = soup.find("title")
        if title_tag:
            result["title"] = title_tag.get_text(strip=True)

        # --- Meta description ---
        meta_desc = (
            soup.find("meta", attrs={"name": "description"}) or
            soup.find("meta", attrs={"property": "og:description"}) or
            soup.find("meta", attrs={"name": "twitter:description"})
        )
        if meta_desc:
            result["meta_description"] = meta_desc.get("content", "")

        # --- Keywords (buat triggered_signals) ---
        meta_kw = soup.find("meta", attrs={"name": "keywords"})
        keywords = meta_kw.get("content", "") if meta_kw else ""

        # Ambil juga OG title sebagai sinyal tambahan
        og_title = soup.find("meta", attrs={"property": "og:title"})
        og_title_text = og_title.get("content", "") if og_title else ""

        result["triggered_signals"] = f"{keywords} {og_title_text}".strip()

        print(f"  [SCRAPE] ✓ Title       : {result['title'][:60] or '-'}")
        print(f"  [SCRAPE] ✓ Meta Desc   : {result['meta_description'][:60] or '-'}")

    except requests.exceptions.Timeout:
        print(f"  [SCRAPE] ✗ Timeout — lanjut pakai URL saja")
    except requests.exceptions.ConnectionError:
        print(f"  [SCRAPE] ✗ Tidak bisa connect — cek URL atau koneksi")
    except requests.exceptions.HTTPError as e:
        print(f"  [SCRAPE] ✗ HTTP Error: {e}")
    except Exception as e:
        print(f"  [SCRAPE] ✗ Error: {e}")

    return result


# ============================================
# TEXT CLEANER (sama dengan train.py)
# ============================================

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9:/._ -]", " ", text)
    return text.strip()

# ============================================
# ENTROPY CALCULATOR
# ============================================

def calculate_entropy(text: str) -> float:
    """Shannon entropy — domain acak biasanya entropy tinggi."""
    if not text:
        return 0.0
    prob = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in prob if p > 0)

# ============================================
# FEATURE EXTRACTOR
# ============================================

def build_features(url: str, title: str = "", meta_description: str = "", triggered_signals: str = "") -> dict:
    """
    Hitung fitur numerik dari URL — IDENTIK dengan train.py.
    Perubahan di sini harus diikuti perubahan di train.py juga.
    """
    parsed   = urlparse(url)
    domain   = parsed.netloc.lower()
    path     = parsed.path.lower()
    query    = parsed.query.lower()
    full_url = url.lower()
    combined = f"{url} {title} {meta_description} {triggered_signals}".lower()

    # Structural
    url_length       = len(url)
    domain_length    = len(domain)
    path_length      = len(path)
    query_length     = len(query)
    dot_count        = full_url.count(".")
    dash_count       = full_url.count("-")
    underscore_count = full_url.count("_")
    digit_count      = sum(c.isdigit() for c in full_url)
    special_chars    = sum(not c.isalnum() for c in full_url)
    is_https         = 1 if url.startswith("https") else 0
    subdomain_count  = max(len(domain.split(".")) - 2, 0)
    has_port         = 1 if ":" in domain else 0
    has_at           = 1 if "@" in url else 0
    tld              = domain.split(".")[-1] if "." in domain else ""

    suspicious_tlds   = {"vip","xyz","top","live","club","online","site","win","bet","casino","poker"}
    is_suspicious_tld = 1 if tld in suspicious_tlds else 0

    # Entropy
    domain_entropy = calculate_entropy(domain)
    path_entropy   = calculate_entropy(path)

    # Keyword hits
    judol_hits_text   = sum(1 for kw in JUDOL_KEYWORDS if kw in combined)
    judol_hits_domain = sum(1 for kw in JUDOL_KEYWORDS if kw in domain or kw in path)
    judol_hits_total  = judol_hits_text + judol_hits_domain

    words       = combined.split()
    total_words = max(len(words), 1)
    keyword_density = judol_hits_total / total_words

    # Repeated words
    word_counts = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1
    repeated_words_count = sum(1 for v in word_counts.values() if v > 1)

    # Ratios
    symbol_ratio   = special_chars / max(url_length, 1)
    digit_ratio    = digit_count   / max(url_length, 1)
    path_url_ratio = path_length   / max(url_length, 1)

    # Suspicious subdomain
    subdomain_parts = domain.split(".")
    suspicious_sub  = sum(
        1 for part in subdomain_parts
        if any(kw in part for kw in SUSPICIOUS_SUBDOMAINS)
    )
    suspicious_subdomain_score = min(suspicious_sub / max(len(subdomain_parts), 1), 1.0)

    # Redirect
    redirect_hits = sum(1 for kw in REDIRECT_INDICATORS if kw in path)
    redirect_indicator_score = min(redirect_hits / max(len(path.split("/")), 1), 1.0)

    # Metadata quality
    metadata_quality = sum([
        1 if title and title not in ("nan", "") else 0,
        1 if meta_description and meta_description not in ("nan", "") else 0,
    ]) / 2.0

    # Judol domain score
    judol_domain_score = min(
        (judol_hits_text + judol_hits_domain * 2) / max(len(JUDOL_KEYWORDS), 1) * 2,
        1.0,
    )

    return {
        "url_length":                url_length,
        "domain_length":             domain_length,
        "path_length":               path_length,
        "query_length":              query_length,
        "dot_count":                 dot_count,
        "dash_count":                dash_count,
        "underscore_count":          underscore_count,
        "digit_count":               digit_count,
        "special_char_count":        special_chars,
        "subdomain_count":           subdomain_count,
        "https":                     is_https,
        "has_port":                  has_port,
        "has_at":                    has_at,
        "is_suspicious_tld":         is_suspicious_tld,
        "domain_entropy":            domain_entropy,
        "path_entropy":              path_entropy,
        "judol_hits_text":           judol_hits_text,
        "judol_hits_domain":         judol_hits_domain,
        "judol_hits_total":          judol_hits_total,
        "keyword_density":           keyword_density,
        "judol_domain_score":        judol_domain_score,
        "repeated_words_count":      repeated_words_count,
        "symbol_ratio":              symbol_ratio,
        "digit_ratio":               digit_ratio,
        "path_url_ratio":            path_url_ratio,
        "suspicious_subdomain_score": suspicious_subdomain_score,
        "redirect_indicator_score":  redirect_indicator_score,
        "metadata_quality":          metadata_quality,
    }


# ============================================
# MODEL LOADER
# ============================================

def load_models():
    print("[INFO] Loading model files...")
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(TFIDF_PATH)
    scaler     = joblib.load(SCALER_PATH)
    print("[SUCCESS] Models loaded\n")
    return model, vectorizer, scaler

# ============================================
# PREDICTOR
# ============================================

def predict(
    url: str,
    title: str = "",
    meta_description: str = "",
    triggered_signals: str = "",
    model  = None,
    vectorizer = None,
    scaler = None,
    verbose: bool = True,
    auto_fetch: bool = True,        # ← otomatis scrape kalau title kosong
) -> dict:
    """
    Prediksi satu URL.
    Kalau auto_fetch=True dan title kosong, otomatis scrape dari URL.
    Returns dict berisi label, confidence, dan probabilitas tiap kelas.
    """

    # Auto scrape kalau title & meta masih kosong
    if auto_fetch and not title and not meta_description:
        scraped = auto_scrape(url)
        title             = scraped["title"]
        meta_description  = scraped["meta_description"]
        triggered_signals = triggered_signals or scraped["triggered_signals"]

    # 1. Buat combined text (sama persis dengan train.py)
    combined = clean_text(
        f"{url} {title} {meta_description} {triggered_signals}"
    )

    # 2. TF-IDF
    X_text = vectorizer.transform([combined])

    # 3. Numerical features — identik dengan train.py
    import pandas as pd
    import warnings

    features = build_features(url, title, meta_description, triggered_signals)

    # Ambil urutan kolom dari scaler (otomatis cocok dengan training)
    try:
        scaler_cols = list(scaler.feature_names_in_)
    except AttributeError:
        scaler_cols = list(features.keys())  # fallback

    num_df = pd.DataFrame(
        [[features.get(col, 0.0) for col in scaler_cols]],
        columns=scaler_cols,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_num_scaled = scaler.transform(num_df)

    # 4. Combine
    X_final = hstack([X_text, X_num_scaled])

    # 5. Predict
    label        = model.predict(X_final)[0]
    proba        = model.predict_proba(X_final)[0]
    class_labels = model.classes_

    prob_dict    = dict(zip(class_labels, proba))
    confidence   = max(proba)

    if verbose:
        print("=" * 55)
        print(" PREDICTION RESULT")
        print("=" * 55)
        print(f"  URL         : {url}")
        print(f"  Title       : {title or '-'}")
        print(f"  Prediction  : {label.upper()}")
        print(f"  Confidence  : {confidence:.2%}")
        print("-" * 55)
        print("  Probability per class:")
        for cls, prob in sorted(prob_dict.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 30)
            print(f"    {cls:<12} {prob:.4f}  {bar}")
        print("-" * 55)
        print("  Key features:")
        print(f"    judol_domain_score : {features['judol_domain_score']:.4f}")
        print(f"    judol_hits_domain  : {int(features['judol_hits_domain'])}")
        print(f"    judol_hits_text    : {int(features['judol_hits_text'])}")
        print(f"    domain_entropy     : {features['domain_entropy']:.4f}")
        print(f"    keyword_density    : {features['keyword_density']:.4f}")
        print(f"    is_suspicious_tld  : {int(features['is_suspicious_tld'])}")
        print("=" * 55)

    return {
        "url":         url,
        "label":       label,
        "confidence":  confidence,
        "probabilities": prob_dict,
        "features":    features,
    }

# ============================================
# BATCH PREDICT
# ============================================

def predict_batch(records: list[dict], model, vectorizer, scaler, auto_fetch: bool = True) -> list[dict]:
    """
    Prediksi banyak URL sekaligus.
    records: list of dict dengan key wajib: url
             key opsional: title, meta_description, triggered_signals
    Kalau auto_fetch=True, title & meta yang kosong akan di-scrape otomatis.
    """
    results = []
    for i, rec in enumerate(records, 1):
        url = rec.get("url", "")
        print(f"\n[{i}/{len(records)}] → {url}")
        result = predict(
            url               = url,
            title             = rec.get("title", ""),
            meta_description  = rec.get("meta_description", ""),
            triggered_signals = rec.get("triggered_signals", ""),
            model             = model,
            vectorizer        = vectorizer,
            scaler            = scaler,
            verbose           = True,
            auto_fetch        = auto_fetch,
        )
        results.append(result)
    return results

# ============================================
# INTERACTIVE MODE
# ============================================

def interactive_mode(model, vectorizer, scaler):
    print("\n" + "=" * 55)
    print("  INTERACTIVE PREDICTION MODE")
    print("  Masukkan URL — title & meta otomatis di-fetch!")
    print("  Ketik 'exit' untuk keluar")
    print("=" * 55)

    while True:
        print()
        url = input("  Masukkan URL : ").strip()
        if url.lower() == "exit":
            print("\n[INFO] Exiting. Bye!")
            break
        if not url:
            print("[WARN] URL tidak boleh kosong.")
            continue

        # Tanya manual override (opsional)
        override = input("  Manual input title & meta? (y/N) : ").strip().lower()
        if override == "y":
            title   = input("  Title        : ").strip()
            meta    = input("  Meta Desc    : ").strip()
            signals = input("  Signals      : ").strip()
            auto    = False
        else:
            title = meta = signals = ""
            auto  = True

        predict(
            url               = url,
            title             = title,
            meta_description  = meta,
            triggered_signals = signals,
            model             = model,
            vectorizer        = vectorizer,
            scaler            = scaler,
            verbose           = True,
            auto_fetch        = auto,
        )

# ============================================
# CLI ENTRYPOINT
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Judol/Malicious URL Predictor",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--url", "-u",
        type=str,
        default=None,
        help="URL yang ingin diprediksi"
    )
    parser.add_argument(
        "--title", "-t",
        type=str,
        default="",
        help="Title halaman (opsional)"
    )
    parser.add_argument(
        "--meta", "-m",
        type=str,
        default="",
        help="Meta description (opsional)"
    )
    parser.add_argument(
        "--signals", "-s",
        type=str,
        default="",
        help="Triggered signals (opsional)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Jalankan mode interaktif (input manual satu per satu)"
    )
    parser.add_argument(
        "--no-scrape",
        action="store_true",
        help="Nonaktifkan auto scraping (pakai title & meta dari argumen saja)"
    )
    parser.add_argument(
        "--batch", "-b",
        type=str,
        default=None,
        help="Path ke CSV untuk batch predict (kolom wajib: url)"
    )

    return parser.parse_args()


def main():
    print("=" * 55)
    print("  JUDOL / MALICIOUS URL DETECTOR")
    print("=" * 55)

    # Load model sekali di awal
    model, vectorizer, scaler = load_models()

    print("  Ketik URL yang ingin dicek.")
    print("  Ketik 'exit' untuk keluar.\n")

    while True:
        url = input("🔗 Masukkan URL : ").strip()

        if url.lower() == "exit":
            print("\n[INFO] Keluar. Bye!")
            break

        if not url:
            print("[WARN] URL tidak boleh kosong, coba lagi.\n")
            continue

        # Tambahkan https:// otomatis kalau lupa
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url
            print(f"[INFO] URL dilengkapi jadi: {url}")

        predict(
            url              = url,
            model            = model,
            vectorizer       = vectorizer,
            scaler           = scaler,
            verbose          = True,
            auto_fetch       = True,   # otomatis scrape title & meta
        )

        print()  # spasi antar prediksi


if __name__ == "__main__":
    main()