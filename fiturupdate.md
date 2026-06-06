**Product Requirements Document (PRD)**

**Fitur: Gambling Ad Detection Engine**

---

### 1. Konteks & Latar Belakang

Web yang sudah lo bangun sebelumnya ingin ditambahkan kemampuan untuk mendeteksi iklan judi online (gambling ads) yang muncul di screenshot halaman website. Deteksi ini berjalan otomatis menggunakan model Computer Vision (Roboflow Workflow dengan SAM3 zero-shot) sehingga tim moderator atau sistem bisa langsung memflagging konten berbahaya tanpa pengecekan manual 100%.

---

### 2. Tujuan

Memberikan fitur baru berupa deteksi otomatis iklan judi online dari screenshot web dengan output berupa:
- Flag boolean (true/false) apakah ada iklan judi.
- Jumlah iklan yang terdeteksi.
- Gambar annotated dengan bounding box untuk review visual.
- Raw data prediksi (koordinat bbox, class name, confidence).

---

### 3. User Stories

- Sebagai moderator, saya ingin mengunggah screenshot website agar sistem otomatis menandai apakah terdapat iklan judi di dalamnya.
- Sebagai sistem/backend, saya ingin menerima flag boolean sehingga bisa memicu aksi auto-block, auto-hide, atau mengirim notifikasi review.
- Sebagai admin, saya ingin melihat history deteksi yang pernah dilakukan lengkap dengan gambar hasil annotasi.

---

### 4. Functional Requirements

**FR-1: Upload & Submit**
- User bisa upload screenshot via drag-drop atau file picker.
- Backend menerima file image (JPG/PNG) dan mengirimkannya ke Roboflow Workflow API.

**FR-2: Deteksi Otomatis**
- Sistem memanggil Roboflow Workflow `Gambling Ad Detector` dengan input image.
- Model mendeteksi class: *gambling advertisement, casino banner, betting ad, judi online banner*.

**FR-3: Parsing Response**
- Backend mengekstrak field berikut dari response API:
  - `gambling_detected` (boolean): true jika ad_count > 0.
  - `ad_count` (integer): total iklan terdeteksi.
  - `output_image` (base64): gambar annotated dengan bounding box.
  - `predictions` (array): detail tiap deteksi (x, y, width, height, class, confidence).

**FR-4: Dashboard / UI Hasil**
- Tampilkan status prominently: **"Safe"** (hijau) atau **"Gambling Ads Detected"** (merah).
- Tampilkan annotated image dengan bounding box.
- Tampilkan jumlah iklan yang terdeteksi.
- Tampilkan daftar prediksi dalam format tabel (class, confidence, koordinat).

**FR-5: History & Logging**
- Simpan setiap hasil deteksi ke database: timestamp, nama file, gambling_detected, ad_count, predictions (JSON), dan annotated image (atau URL-nya).
- Tersedia halaman history dengan filter dan pagination.

**FR-6: Auto-Action (Opsional / Future)**
- Jika `gambling_detected` = true, sistem bisa memicu webhook, mengirim email alert, atau menandai entry sebagai "needs review".

---

### 5. Non-Functional Requirements

**NFR-1: Performance**
- Latency end-to-end (upload sampai hasil muncul): target di bawah 5 detik untuk gambar standar (1920x1080 ke bawah) via Roboflow Cloud API.
- Jika traffic tinggi, pertimbangkan queue atau rate limiting.

**NFR-2: Akurasi**
- Menggunakan zero-shot SAM3, sehingga akurasi bergantung pada kualitas prompt dan variasi visual iklan.
- Diperkirakan cukup untuk MVP. Jika false positive/negative tinggi, rencanakan iterasi ke custom model training.

**NFR-3: Security**
- API Key Roboflow disimpan di environment variable, bukan hardcode di client-side.
- Validasi file upload (type, size max 10MB).

**NFR-4: Scalability**
- Endpoint API Roboflow dipanggil dari backend, bukan dari browser, agar API Key tidak expose.

---

### 6. Integrasi Teknis & Data Flow

**Arsitektur Singkat:**
```
Frontend (Upload) 
    -> Backend Server (Python/Node/PHP)
        -> Roboflow Workflow API (Cloud)
    <- Backend Parsing & Save to DB
<- Frontend (Render Hasil + History)
```

**API Contract (Backend ke Roboflow):**
- **SDK:** `inference-sdk` (Python) atau HTTP POST langsung.
- **Input:** `image` (file path atau base64).
- **Output:** 
  - `output_image`: base64 annotated image.
  - `gambling_detected`: boolean.
  - `ad_count`: integer.
  - `predictions`: array of detections.

**Contoh Snippet Integrasi (Python):**
```python
from inference_sdk import InferenceHTTPClient
import base64

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="P5mMYO1Mp0ROjcv6yg8S"
)

result = client.run_workflow(
    workspace_name="15240133s-workspace",
    workflow_id="gambling-ad-detector-1779947644260",
    images={"image": "screenshot.png"},
    use_cache=True
)

output = result[0]
flag = output["gambling_detected"]
count = output["ad_count"]
predictions = output["predictions"]
annotated_b64 = output["output_image"]
```

---

### 7. UI/UX Requirements

- **Upload Zone:** Area drag-drop dengan preview thumbnail.
- **Result Card:** 
  - Badge status besar (Safe / Detected).
  - Counter jumlah iklan.
  - Preview annotated image (dapat di-zoom).
- **Prediction Table:** Kolom Class, Confidence (%), Koordinat.
- **History Page:** Tabel dengan kolom Tanggal, Status, Jumlah Iklan, Thumbnail, Action (View Detail).

---

### 8. Success Metrics / KPI

- **Precision MVP:** Dari 50 screenshot uji coba, berapa persen yang flag-nya sesuai ekspektasi manusia.
- **Response Time:** Rata-rata waktu proses dari upload sampai hasil ditampilkan.
- **Adoption Rate:** Jumlah screenshot yang diproses per minggu setelah fitur rilis.
- **False Positive Rate:** Persentase screenshot "aman" yang salah flag sebagai gambling.

---

### 9. Risiko & Mitigasi

| Risiko | Mitigasi |
|--------|----------|
| SAM3 zero-shot miss pada iklan dengan style tidak umum | Iterasi prompt class names. Jika masih kurang, alihkan ke custom training (Rapid / labeled dataset). |
| Cost API call meningkat | Implementasi rate limit per user dan cache untuk gambar identik. |
| Latency lambat karena ukuran screenshot besar | Compress/resize image sebelum kirim ke API jika perlu. |

---

### 10. Timeline (Estimasi Kasar)

| Task | Estimasi |
|------|----------|
| Setup backend client & test API endpoint | 1 hari |
| Endpoint upload + parsing response | 1 hari |
| UI upload + result preview | 2 hari |
| History page + DB schema | 2 hari |
| Testing dengan sample screenshot real | 1 hari |

**Total:** Sekitar 5-7 hari kerja untuk MVP.

---

### 11. Open Questions

- Apakah perlu fitur auto-capture screenshot dari URL website (bukan upload manual)?
- Apakah perlu klasifikasi tambahan (misal, bedakan jenis judi: slot, sportsbook, togel)?
- Apakah perlu mode batch processing untuk banyak screenshot sekaligus?

---

Kalau lo butuh bantuan translate ke bahasa Inggris untuk tim internasional, atau mau gw bantu buat skema database dan API endpoint spesifik buat backend lo, bilang aja.