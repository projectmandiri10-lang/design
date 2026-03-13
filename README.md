# AI Screen Printing Vectorizer

Perangkat lunak lokal untuk mengubah foto artwork sablon kaos menjadi vektor SVG yang bersih untuk kebutuhan screen printing. Seluruh stack berjalan secara lokal: frontend React + Vite, backend Express, dan mesin pemrosesan gambar Python.

## Kebutuhan

- Windows 10/11
- Node.js LTS dengan `npm`
- Python 3.11
- Potrace terpasang dan tersedia di `PATH`
- Opsional: bobot model Real-ESRGAN di `models/RealESRGAN_x4plus.pth`

## Struktur Proyek

- `frontend` - antarmuka React + Vite + Tailwind
- `backend` - API Express dan jembatan worker Python
- `ai-engine` - pipeline Python dan modul tiap tahap
- `models` - bobot model lokal
- `uploads` - file upload sementara
- `outputs` - preview dan file SVG hasil

## Setup di Windows

### 1. Install Node.js

- Install Node.js LTS terbaru yang sudah menyertakan `npm`.
- Verifikasi dengan:
  - `node -v`
  - `npm -v`

### 2. Install Python 3.11

- Install Python 3.11 berdampingan dengan versi Python lain jika diperlukan.
- Buat virtual environment dari root repo:
  - `py -3.11 -m venv .venv`
  - `.\\.venv\\Scripts\\Activate.ps1`

### 3. Install Potrace

- Install binary Potrace untuk Windows dan pastikan `potrace --version` bisa dijalankan dari PowerShell.
- Jika Potrace tidak ada di `PATH`, atur `POTRACE_BIN` saat menjalankan backend.
- Alternatif lokal untuk project ini: letakkan `potrace.exe` di `tools/potrace/potrace-1.16.win64/potrace.exe` dan backend akan mendeteksinya otomatis.

### 4. Install Dependensi Python

- Aktifkan virtual environment Python 3.11.
- Jalankan:
  - `python -m pip install --upgrade pip`
  - `python -m pip install -r ai-engine/requirements.txt`

### 5. Unduh Model Real-ESRGAN

- Jalankan script resmi repo untuk mengunduh model:
  - `.\download-model.ps1`
- Jika ingin memaksa unduh ulang:
  - `.\download-model.ps1 -Force`
- File bobot `RealESRGAN_x4plus` akan ditempatkan di folder `models/`.
- Path yang diharapkan:
  - `models/RealESRGAN_x4plus.pth`

Jika file model tidak tersedia, pipeline akan memakai mode upscale klasik dan melaporkannya di respons API.

### 6. Install Dependensi JavaScript

- Dari root repo:
  - `npm install`
- Salin template env bila diperlukan:
  - `Copy-Item backend/.env.example backend/.env`
  - `Copy-Item frontend/.env.example frontend/.env`

## Menjalankan Aplikasi

### Terminal 1 - Backend

- Aktifkan environment Python 3.11.
- Jalankan API:
  - `npm run dev --workspace backend`

Variabel environment:

- `PORT` - port backend, default `3001`
- `PYTHON_EXECUTABLE` - path Python, default `python`
- `POTRACE_BIN` - nama binary Potrace atau path absolut
- `MODEL_DIR` - direktori model, default `models`
- `OUTPUT_RETENTION_HOURS` - umur file sebelum dibersihkan, default `24`

### Terminal 2 - Frontend

- Jalankan UI Vite:
  - `npm run dev --workspace frontend`

Variabel environment opsional:

- `VITE_API_BASE_URL` - URL backend, default `http://localhost:3001`

### Bantuan opsional

- `start-local.bat` dapat di-double-click untuk membuka terminal backend dan frontend secara otomatis.
- `download-model.ps1` mengunduh bobot resmi `RealESRGAN_x4plus.pth` ke folder `models/`.

## API

### `POST /api/vectorize`

Field multipart form:

- `image` - file input `jpg`, `png`, atau `webp`
- `colorCount` - integer `2` sampai `6`

Respons:

- `jobId`
- `svgContent`
- `svgFile`
- `processedPreviewFile`
- `originalFile`
- `palette`
- `timings`
- `warnings`
- `fallbacks`

### `GET /api/files/:jobId/:name`

Menyajikan file hasil yang berada di folder output sesuai job terkait.

## Catatan

- Job diproses satu per satu agar stabil di Windows CPU-only dengan RAM 8 GB.
- Hasil akhir berupa satu dokumen SVG yang memuat semua warna dalam satu file.
- Potrace tetap menjadi tracer utama; `svgpathtools` menangani perapian path secara deterministik, dan integrasi DiffVG bersifat opsional saat runtime.
