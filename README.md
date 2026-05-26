# Web Checker Premium

Deskripsi
-	Web Checker Premium adalah aplikasi berbasis Flask untuk memeriksa dan menampilkan hasil pemeriksaan (checker) terhadap konten atau layanan tertentu. Proyek ini tampak menyertakan modul ML untuk analisis, halaman web untuk otentikasi dan pembayaran, serta penyimpanan sesi Flask.

Fitur
- Aplikasi web berbasis Flask
- Halaman otentikasi: login, register, profile
- Halaman pembayaran dan verifikasi: payment, auth-payment, success
- Halaman hasil pemeriksaan: result
- Integrasi folder `ML/` dan `training.py` untuk kebutuhan machine learning

Persyaratan
- Python 3.8+ (direkomendasikan)
- Dependensi tercantum di `requirements.txt`

Instalasi (Windows / PowerShell)
```powershell
# Buat virtual environment
python -m venv venv
# Aktifkan venv
venv\Scripts\Activate.ps1
# Pasang dependensi
pip install -r requirements.txt
```

Menjalankan aplikasi (development)
```powershell
# Dari root proyek
python app.py
# atau jika app.py menggunakan FLASK_APP
# set FLASK_APP=app.py
# flask run
```

Setup database
- File `database.sql` berisi schema atau data contoh. Import ke SQLite/MySQL/Postgres sesuai kebutuhan (sesuaikan konfigurasi di `app.py` atau file konfigurasi lain jika ada).

Struktur Proyek (inti)
- app.py
- requirements.txt
- database.sql
- training.py
- ML/  (folder ML untuk model atau skrip terkait)
- static/
  - css/style.css
  - img/
- templates/
  - home.html
  - login.html
  - register.html
  - dashboard.html
  - profile.html
  - payment.html
  - auth-payment.html
  - result.html
  - success.html
- flask_session/ (folder sesi Flask, berisi file sesi)

Penggunaan singkat
1. Daftar atau login melalui halaman `register.html` / `login.html`.
2. Akses dashboard untuk melakukan pemeriksaan atau mengelola data.
3. Gunakan halaman `payment.html` untuk proses pembayaran (jika diaktifkan).
4. Hasil pemeriksaan ditampilkan pada `result.html`.

Catatan development
- Jika Anda mengembangkan bagian ML, cek `training.py` dan folder `ML/`.
- Folder `flask_session/` berisi sesi yang mungkin dibuat saat menjalankan aplikasi. Jangan masukkan folder ini ke VCS jika berisi data sensitif.

Kontribusi
- Buka issue atau fork repo lalu kirim pull request untuk kontribusi.

Lisensi
- Tambahkan lisensi sesuai kebutuhan (mis. MIT) — file lisensi belum disertakan.

Kontak
- Untuk pertanyaan atau bantuan: tambahkan detail kontak atau buat issue di repositori.

---

Terima kasih telah menggunakan Web Checker Premium!
