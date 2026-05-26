import os
import time
import random
import re
import math
import joblib
import numpy as np
import requests
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_session import Session
import pymysql
from werkzeug.security import generate_password_hash, check_password_hash
import midtransclient
from dotenv import load_dotenv
from functools import wraps
from bs4 import BeautifulSoup
from scipy.sparse import hstack
from urllib.parse import urlparse

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# Konfigurasi Session untuk disimpan di server-side agar lebih aman
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# =============================================
# ML MODEL LOADING
# =============================================
ML_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ML', 'model.pkl')
ML_TFIDF_PATH = os.path.join(os.path.dirname(__file__), 'ML', 'tfidf.pkl')
ML_SCALER_PATH = os.path.join(os.path.dirname(__file__), 'ML', 'scaler.pkl')

ml_model = None
ml_vectorizer = None
ml_scaler = None

def load_ml_models():
    global ml_model, ml_vectorizer, ml_scaler
    try:
        print("[ML] Loading model files...")
        ml_model = joblib.load(ML_MODEL_PATH)
        ml_vectorizer = joblib.load(ML_TFIDF_PATH)
        ml_scaler = joblib.load(ML_SCALER_PATH)
        print("[ML] Models loaded successfully!")
        return True
    except Exception as e:
        print(f"[ML] Error loading models: {e}")
        return False

# Load models on startup
load_ml_models()

# =============================================
# ML PREDICTION CONFIG & FUNCTIONS
# =============================================
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

SCRAPER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9:/._ -]", " ", text)
    return text.strip()

def calculate_entropy(text: str) -> float:
    if not text:
        return 0.0
    prob = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in prob if p > 0)

def auto_scrape(url: str, timeout: int = 5) -> dict:
    """Fetch title & meta dari URL"""
    result = {
        "title": "",
        "meta_description": "",
        "triggered_signals": "",
    }
    try:
        resp = requests.get(url, headers=SCRAPER_HEADERS, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        title_tag = soup.find("title")
        if title_tag:
            result["title"] = title_tag.get_text(strip=True)
        
        meta_desc = (
            soup.find("meta", attrs={"name": "description"}) or
            soup.find("meta", attrs={"property": "og:description"})
        )
        if meta_desc:
            result["meta_description"] = meta_desc.get("content", "")
    except Exception as e:
        print(f"[SCRAPE] Error: {e}")
    
    return result

def build_ml_features(url: str, title: str = "", meta_description: str = "", triggered_signals: str = "") -> dict:
    """Build ML features identik dengan training.py"""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    query = parsed.query.lower()
    full_url = url.lower()
    combined = f"{url} {title} {meta_description} {triggered_signals}".lower()

    # Structural
    url_length = len(url)
    domain_length = len(domain)
    path_length = len(path)
    dot_count = full_url.count(".")
    dash_count = full_url.count("-")
    digit_count = sum(c.isdigit() for c in full_url)
    special_chars = sum(not c.isalnum() for c in full_url)
    is_https = 1 if url.startswith("https") else 0
    tld = domain.split(".")[-1] if "." in domain else ""
    suspicious_tlds = {"vip", "xyz", "top", "live", "club", "online", "site", "win", "bet", "casino", "poker"}
    is_suspicious_tld = 1 if tld in suspicious_tlds else 0

    # Entropy
    domain_entropy = calculate_entropy(domain)

    # Keyword hits
    judol_hits_text = sum(1 for kw in JUDOL_KEYWORDS if kw in combined)
    judol_hits_domain = sum(1 for kw in JUDOL_KEYWORDS if kw in domain or kw in path)
    judol_hits_total = judol_hits_text + judol_hits_domain

    words = combined.split()
    total_words = max(len(words), 1)
    keyword_density = judol_hits_total / total_words

    # Repeated words
    word_counts = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1
    repeated_words_count = sum(1 for v in word_counts.values() if v > 1)

    # Ratios
    symbol_ratio = special_chars / max(url_length, 1)

    # Suspicious subdomain
    subdomain_parts = domain.split(".")
    suspicious_sub = sum(1 for part in subdomain_parts if any(kw in part for kw in SUSPICIOUS_SUBDOMAINS))
    suspicious_subdomain_score = min(suspicious_sub / max(len(subdomain_parts), 1), 1.0)

    # Redirect
    redirect_hits = sum(1 for kw in REDIRECT_INDICATORS if kw in path)
    redirect_indicator_score = min(redirect_hits / max(len(path.split("/")), 1), 1.0)

    # Metadata quality
    metadata_quality = sum([
        1 if title and title not in ("nan", "") else 0,
        1 if meta_description and meta_description not in ("nan", "") else 0,
    ]) / 2.0

    return {
        "url_length": url_length,
        "domain_length": domain_length,
        "path_length": path_length,
        "dot_count": dot_count,
        "dash_count": dash_count,
        "digit_count": digit_count,
        "special_char_count": special_chars,
        "https": is_https,
        "is_suspicious_tld": is_suspicious_tld,
        "domain_entropy": domain_entropy,
        "judol_hits_text": judol_hits_text,
        "judol_hits_domain": judol_hits_domain,
        "judol_hits_total": judol_hits_total,
        "keyword_density": keyword_density,
        "repeated_words_count": repeated_words_count,
        "symbol_ratio": symbol_ratio,
        "suspicious_subdomain_score": suspicious_subdomain_score,
        "redirect_indicator_score": redirect_indicator_score,
        "metadata_quality": metadata_quality,
    }

def predict_url(url: str) -> dict:
    """Prediksi malicious/judol URL menggunakan ML model"""
    if not all([ml_model, ml_vectorizer, ml_scaler]):
        return {"error": "ML models not loaded"}
    
    try:
        # Auto scrape
        scraped = auto_scrape(url)
        title = scraped["title"]
        meta_description = scraped["meta_description"]
        
        # Clean combined text
        combined = clean_text(f"{url} {title} {meta_description}")
        
        # TF-IDF
        X_text = ml_vectorizer.transform([combined])
        
        # Build numerical features
        features = build_ml_features(url, title, meta_description, "")
        
        # Get scaler columns
        try:
            scaler_cols = list(ml_scaler.feature_names_in_)
        except AttributeError:
            scaler_cols = list(features.keys())
        
        # Create DataFrame dengan kolom yang tepat
        num_df = pd.DataFrame(
            [[features.get(col, 0.0) for col in scaler_cols]],
            columns=scaler_cols,
        )
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_num_scaled = ml_scaler.transform(num_df)
        
        # Combine
        X_final = hstack([X_text, X_num_scaled])
        
        # Predict
        label = ml_model.predict(X_final)[0]
        proba = ml_model.predict_proba(X_final)[0]
        class_labels = ml_model.classes_
        
        prob_dict = dict(zip(class_labels, proba))
        confidence = max(proba)
        
        return {
            "url": url,
            "label": label,
            "confidence": float(confidence),
            "probabilities": {str(k): float(v) for k, v in prob_dict.items()},
            "title": title,
            "meta_description": meta_description,
            "features": features,
        }
    except Exception as e:
        print(f"[ML Prediction Error]: {e}")
        return {
            "url": url,
            "label": "unknown",
            "confidence": 0.0,
            "probabilities": {},
            "error": str(e),
        }

def normalize_prediction_label(label) -> str:
    label_text = str(label).strip().lower()
    if not label_text:
        return "unknown"

    malicious_tokens = {"malicious", "danger", "unsafe", "phishing", "phish", "spam", "judol", "bad"}
    safe_tokens = {"safe", "benign", "clean", "normal", "legit", "ham"}

    if any(token in label_text for token in malicious_tokens):
        return "malicious"
    if any(token in label_text for token in safe_tokens):
        return "safe"
    return label_text

def build_actionable_insights(url: str, features: dict, label: str, confidence: float) -> dict:
    """Turn ML output into practical findings, anomalies, and next actions."""
    features = features or {}
    label = normalize_prediction_label(label)

    indicators = []
    vulnerabilities = []
    anomalies = []
    recommendations = []

    domain_entropy = float(features.get("domain_entropy", 0.0) or 0.0)
    keyword_density = float(features.get("keyword_density", 0.0) or 0.0)
    judol_hits_total = int(features.get("judol_hits_total", 0) or 0)
    suspicious_tld = int(features.get("is_suspicious_tld", 0) or 0)
    suspicious_subdomain_score = float(features.get("suspicious_subdomain_score", 0.0) or 0.0)
    redirect_indicator_score = float(features.get("redirect_indicator_score", 0.0) or 0.0)
    metadata_quality = float(features.get("metadata_quality", 0.0) or 0.0)
    https_enabled = int(features.get("https", 0) or 0)
    url_length = int(features.get("url_length", 0) or 0)

    risk_score = 0

    if label == "malicious":
        risk_score += 35
        indicators.append("Model classifies this URL as malicious")
    elif label == "safe":
        risk_score += 5
        indicators.append("Model classifies this URL as safe")
    else:
        risk_score += 20
        indicators.append("Model output is inconclusive")

    if confidence >= 0.85:
        risk_score += 15
    elif confidence >= 0.65:
        risk_score += 8
    else:
        risk_score += 2

    if judol_hits_total > 0:
        risk_score += min(20, judol_hits_total * 4)
        indicators.append(f"{judol_hits_total} gambling-related keyword hit(s) found")
        vulnerabilities.append({
            "title": "Judol Keyword Match",
            "severity": "HIGH" if judol_hits_total >= 2 else "MEDIUM",
            "domain": "Content Signals",
            "description": "Page content or URL contains gambling/judol indicators that often appear on malicious pages.",
        })

    if suspicious_tld:
        risk_score += 10
        indicators.append("Suspicious top-level domain detected")
        vulnerabilities.append({
            "title": "Suspicious TLD",
            "severity": "MEDIUM",
            "domain": "Domain Reputation",
            "description": "The domain extension belongs to a TLD that is frequently abused by scam or spam infrastructure.",
        })

    if suspicious_subdomain_score > 0:
        risk_score += min(12, int(suspicious_subdomain_score * 20))
        indicators.append("Suspicious subdomain pattern detected")
        vulnerabilities.append({
            "title": "Suspicious Subdomain Pattern",
            "severity": "MEDIUM",
            "domain": "Host Structure",
            "description": "Subdomain structure contains terms frequently used for lure, redirect, or fake-login pages.",
        })

    if redirect_indicator_score > 0:
        risk_score += min(10, int(redirect_indicator_score * 20))
        indicators.append("Redirect-like path detected")
        vulnerabilities.append({
            "title": "Redirect Behavior",
            "severity": "MEDIUM",
            "domain": "URL Routing",
            "description": "Path patterns suggest the page may redirect users through tracking or affiliate chains.",
        })

    if domain_entropy >= 3.7:
        risk_score += 8
        indicators.append("High domain entropy detected")
        vulnerabilities.append({
            "title": "High Domain Entropy",
            "severity": "LOW" if domain_entropy < 4.0 else "MEDIUM",
            "domain": "Domain Structure",
            "description": "Random-looking domain strings are often used to rotate malicious infrastructure.",
        })

    if metadata_quality < 1.0:
        risk_score += 6
        indicators.append("Missing or weak metadata")
        vulnerabilities.append({
            "title": "Low Metadata Quality",
            "severity": "LOW",
            "domain": "Page Metadata",
            "description": "The page exposes limited metadata, reducing trust signals for users and scanners.",
        })

    if not https_enabled:
        risk_score += 10
        indicators.append("HTTPS is not enabled")
        vulnerabilities.append({
            "title": "No HTTPS",
            "severity": "HIGH",
            "domain": "Transport Security",
            "description": "The URL does not use HTTPS, so traffic can be intercepted or modified in transit.",
        })

    if url_length >= 120:
        risk_score += 6
        indicators.append("Very long URL detected")

    if keyword_density >= 0.15:
        risk_score += 8
        indicators.append("Keyword density is unusually high")

    if label == "malicious" and risk_score >= 70:
        risk_level = "critical"
    elif risk_score >= 55:
        risk_level = "high"
    elif risk_score >= 30:
        risk_level = "medium"
    else:
        risk_level = "low"

    if "malicious" in label:
        anomalies.extend([
            {
                "url": f"{url.rstrip('/')}/admin",
                "risk": "Admin surface exposed",
            },
            {
                "url": f"{url.rstrip('/')}/login",
                "risk": "Credential harvesting pattern",
            },
        ])

    if redirect_indicator_score > 0:
        anomalies.append({
            "url": f"{url}?redirect=external",
            "risk": "Open redirect / tracking indicator",
        })

    if not https_enabled:
        anomalies.append({
            "url": url,
            "risk": "Traffic transmitted without TLS",
        })

    if judol_hits_total > 0:
        anomalies.append({
            "url": url,
            "risk": "Judol/gambling keyword surface",
        })

    recommendations = [
        "Block the URL at the gateway or browser protection layer until reviewed.",
        "Verify the domain ownership and certificate status before allowing access.",
        "Search the page source for redirect chains, affiliate links, or credential capture forms.",
    ]

    if not https_enabled:
        recommendations.insert(0, "Enable HTTPS and renew the TLS certificate immediately.")
    if label == "malicious":
        recommendations.insert(0, "Treat the page as unsafe and avoid submitting any credentials or tokens.")

    return {
        "risk_score": min(risk_score, 100),
        "risk_level": risk_level,
        "indicators": indicators[:6],
        "vulnerabilities": vulnerabilities[:5],
        "anomalies": anomalies[:6],
        "recommendations": recommendations[:4],
    }

# Fungsi helper koneksi database MySQL
def get_db_connection():
    return pymysql.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME'),
        cursorclass=pymysql.cursors.DictCursor
    )

# Inisialisasi Midtrans Snap Client
snap = midtransclient.Snap(
    is_production=False,
    server_key=os.getenv('MIDTRANS_SERVER_KEY'),
    client_key=os.getenv('MIDTRANS_CLIENT_KEY')
)

# --- MIDDLEWARE / DECORATORS ---
def ensure_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            url_param = request.args.get('url')
            if url_param:
                session['redirect_to'] = url_for('home', url=url_param)
            else:
                session['redirect_to'] = request.url
            
            flash("Silakan login terlebih dahulu untuk mengakses fitur ini.", "error")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def ensure_premium(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' in session and session['user']['is_premium'] == 1:
            return f(*args, **kwargs)
        
        flash("Fitur Cek Website membutuhkan akses Premium. Silakan lakukan pembayaran.", "warning")
        return redirect(url_for('payment'))
    return decorated_function


# ==========================================
# 3. BAGIAN ROUTE (Sudah Diperbaiki)
# ==========================================

@app.route('/')
@app.route('/dashboard')
def dashboard():
    # Menampilkan halaman landing / dashboard awal (Satu fungsi untuk 2 rute agar tidak duplikat)
    current_user = session.get('user')
    return render_template('dashboard.html', user=current_user)

@app.route('/home')
def home():
    # Ambil data user yang disimpan saat login dari session
    current_user = session.get('user') 
    return render_template('home.html', user=current_user)

# Pasang @ensure_auth dan @ensure_premium tepat di atas fungsi check_website
@app.route('/check-website')
@ensure_auth       # <--- Mengecek Login pertama kali
@ensure_premium    # <--- Mengecek Premium setelahnya
def check_website():
    target_url = request.args.get('url', '')
    current_user = session.get('user')
    
    # Tambah https:// otomatis kalau belum ada
    if target_url and not target_url.startswith('http'):
        target_url = 'https://' + target_url
    
    # Gunakan ML model untuk prediksi
    ml_result = predict_url(target_url) if target_url else {}
    normalized_label = normalize_prediction_label(ml_result.get('label', 'unknown'))
    insights = build_actionable_insights(
        target_url,
        ml_result.get('features', {}),
        normalized_label,
        float(ml_result.get('confidence', 0.0) or 0.0),
    )
    risk_score = insights['risk_score']
    
    # Build result dengan data dari ML
    result = {
        'url': target_url,
        'ml_label': normalized_label,
        'ml_confidence': ml_result.get('confidence', 0.0),
        'ml_probabilities': ml_result.get('probabilities', {}),
        'title': ml_result.get('title', '-'),
        'meta_description': ml_result.get('meta_description', '-'),
        'features': ml_result.get('features', {}),
        'risk_score': risk_score,
        'risk_level': insights['risk_level'],
        'analysis_indicators': insights['indicators'],
        'vulnerabilities': insights['vulnerabilities'],
        'anomalies': insights['anomalies'],
        'recommendations': insights['recommendations'],
        'seoScore': max(0, 100 - risk_score),
        'loadTime': round(0.8 + (len(target_url) / 120 if target_url else 0), 2),
    }
    
    return render_template('result.html', result=result, user=current_user)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email_or_username = request.form['emailOrUsername']
        password = request.form['password']
        
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                # Cek apakah username atau email ada di database
                cursor.execute(
                    "SELECT * FROM users WHERE email = %s OR username = %s", 
                    (email_or_username, email_or_username)
                )
                user = cursor.fetchone()
                
                # JIKA USER BELUM TERDAFTAR (BELUM REGISTER)
                if not user:
                    flash("Akun belum terdaftar! Silakan buat akun baru terlebih dahulu.", "info")
                    return redirect(url_for('register'))
                
                # Perbaikan pengambilan data variabel session menggunakan objek 'user' hasil fetch database
                if check_password_hash(user['password'], password):
                    session['user'] = {
                        'id': user['id'],
                        'username': user['username'],
                        'email': user['email'],
                        'is_premium': user['is_premium']
                    }
                    
                    redirect_to = session.pop('redirect_to', url_for('home'))
                    return redirect(redirect_to)
                else:
                    error = "Password yang Anda masukkan salah!"
        finally:
            conn.close()
            
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                # Cek apakah email sudah terdaftar
                cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
                if cursor.fetchone():
                    error = "Email sudah terdaftar!"
                else:
                    # Hash password dan masukkan data user baru
                    hashed_pwd = generate_password_hash(password)
                    cursor.execute(
                        "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                        (username, email, hashed_pwd)
                    )
                    conn.commit()
                    
                    flash("Registrasi berhasil! Silakan login melalui menu di atas.", "success")
                    return redirect(url_for('login'))
        finally:
            conn.close()
            
    return render_template('register.html', error=error)

@app.route('/kembali')
def kembali():
    if 'user' in session:
        return redirect(url_for('home'))
    else:
        return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    # 1. Hapus session 'user'
    session.pop('user', None) 
    session.clear()

    # 2. Kembalikan user ke halaman dashboard sesuai permintaan Anda
    return redirect(url_for('dashboard'))

@app.route('/payment')
@ensure_auth
def payment():
    if session['user']['is_premium'] == 1:
        return redirect(url_for('home'))
    return render_template('payment.html', user=session['user'], clientKey=os.getenv('MIDTRANS_CLIENT_KEY'))

@app.route('/payment/token', methods=['POST'])
@ensure_auth
def payment_token():
    data = request.json or {}
    chosen_package = data.get('package', 'basic')
    
    package_prices = {
        'basic': 25000,
        'pro': 50000,
        'custom': 150000
    }
    
    amount = package_prices.get(chosen_package, 25000)
    order_id = f"{chosen_package.upper()}-{int(time.time())}-{session['user']['id']}"
    
    param = {
        "transaction_details": {
            "order_id": order_id,
            "gross_amount": amount
        },
        "credit_card": {"secure": True},
        "enabled_payments": [
            "credit_card", "bca_va", "bni_va", "bri_va", "mandiri_va", 
            "permata_va", "gopay", "shopeepay", "qris", "alfamart", "indomaret"
        ],
        "customer_details": {
            "username": session['user']['username'],
            "email": session['user']['email']
        }
    }
    
    try:
        transaction = snap.create_transaction(param)
        return jsonify({"token": transaction['token']})
    except Exception as e:
        print(e)
        return jsonify({"error": "Gagal membuat token pembayaran"}), 500

@app.route('/payment/success', methods=['POST'])
@ensure_auth
def payment_success_handler():
    data = request.json
    user_id = session['user']['id']
    order_id = data.get('order_id', '')
    
    package_type = order_id.split('-')[0].lower() if order_id else 'basic'
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """INSERT INTO payments (user_id, transaction_id, package_type, payment_type, gross_amount, transaction_status) 
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (user_id, data['transaction_id'], package_type, data['payment_type'], data['gross_amount'], data['transaction_status'])
            )
            
            cursor.execute(
                "UPDATE users SET is_premium = 1, package_type = %s WHERE id = %s", 
                (package_type, user_id)
            )
            conn.commit()
            
            session['user']['is_premium'] = 1
            session['user']['package_type'] = package_type
            session.modified = True
            
            return jsonify({"status": "success"})
    except Exception as e:
        print(e)
        return jsonify({"error": "Gagal memperbarui status premium"}), 500
    finally:
        conn.close()

@app.route('/payment-success')
@ensure_auth
def payment_success_page():
    current_user = session.get('user')
    return render_template('success.html', user=current_user)

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        flash(f"Link reset password telah dikirim ke {email} (Simulasi)", "info")
        return redirect(url_for('login'))
        
    return "<h3>Halaman Fitur Reset Password (Dapat dikembangkan menggunakan SMTP Email)</h3>"   

@app.route('/profile')
def profile():
    current_user = session.get('user') 
    if not current_user:
        return redirect(url_for('login'))
        
    return render_template('profile.html', user=current_user) 

if __name__ == '__main__':
    app.run(port=3000, debug=True)