# AUTO VECTOR SABLON AI

Desktop Windows application to convert a photo of a printed t-shirt design into a clean screen-print-ready SVG.

The app runs locally and uses:

- Stable Diffusion WebUI (AUTOMATIC1111) for AI cleanup
- ControlNet for lineart-based outline preservation
- Potrace for bitmap-to-SVG tracing
- Inkscape for optional EPS/PDF export

## Main Workflow

1. Import Image
2. Clean AI
3. Detect Outline
4. Reduce Colors
5. Generate Vector
6. Export SVG

The main UI is prompt-centric and step-based. Advanced features from the previous app are still available from the top toolbar and the `Advanced Settings` dock.

## Project Structure

```text
auto_vector_sablon_ai/
  app.py
  config.py
  ui_main.py
  main.py
  modules/
    preprocess.py
    ai_cleanup.py
    edge_detect.py
    color_reduce.py
    vectorize.py
    export_svg.py
  core/
  ui/
  export/
  assets/
  models/
  output/
  temp/
  tests/
```

## Technology Stack

- Python 3.10+
- PySide6
- OpenCV
- NumPy
- Pillow
- requests
- scikit-learn
- svgwrite
- ONNX Runtime

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Untuk pengguna Windows yang ingin paling sederhana:

1. Jalankan `setup.bat` sekali.
2. Setelah setup selesai, jalankan `run_app.bat`.
3. Jika ingin fitur AI aktif, nyalakan AUTOMATIC1111 WebUI dengan `--api` lebih dulu.

Install external tools as needed:

- Potrace in `PATH`, `POTRACE_BIN`, or `assets/tools/potrace/.../potrace.exe`
- Inkscape in `PATH` or `INKSCAPE_BIN`
- AUTOMATIC1111 WebUI with `--api` enabled if you want AI cleanup and ControlNet

## Run

```powershell
.\run_app.bat
```

or:

```powershell
python app.py
```

`run_app.bat` sekarang akan:

- membuat virtual environment otomatis jika belum ada
- install dependency otomatis jika masih kurang
- memberi peringatan jika AUTOMATIC1111 belum aktif di `127.0.0.1:7860`

Jika Anda melihat pesan berikut di dalam aplikasi:

- `Clean AI fallback: requests is not installed.`
  Jalankan `setup.bat` agar dependency Python terpasang.
- `Outline fallback: Failed to connect to A1111 at http://127.0.0.1:7860`
  Artinya AUTOMATIC1111 belum berjalan atau belum memakai `--api`.

## Panduan A1111 untuk Windows AMD

Jika Anda memakai GPU AMD di Windows, dokumentasi resmi AUTOMATIC1111 menyebut dukungan Windows+AMD belum resmi di repo utama, dan menyarankan fork DirectML. Untuk aplikasi ini, tujuan minimumnya adalah:

1. WebUI bisa dibuka.
2. API aktif di `http://127.0.0.1:7860`.
3. Extension ControlNet terpasang.
4. Model `control_v11p_sd15_lineart` tersedia.

Per 16 Maret 2026, ukuran download utama yang perlu Anda siapkan kira-kira:

- Python 3.10.6 Windows installer 64-bit: 27.6 MB
- Git for Windows installer x64: sekitar 61.58 MB
- Checkpoint Stable Diffusion 1.5 contoh `v1-5-pruned-emaonly.safetensors`: 4.27 GB
- ControlNet model `control_v11p_sd15_lineart.pth`: 1.45 GB
- Alternatif model ControlNet ukuran sedang menurut wiki resmi: 723 MB

Praktisnya, total download untuk jalur AI biasanya di atas 6 GB, belum termasuk paket Python yang akan diunduh otomatis saat A1111 pertama kali dijalankan.

Langkah ringkas:

1. Install Python 3.10.6 dan centang `Add Python to PATH`.
2. Install Git for Windows.
3. Buat folder misalnya `C:\AI`.
4. Buka Command Prompt di folder itu lalu jalankan:

```bat
git clone https://github.com/lshqqytiger/stable-diffusion-webui-directml
cd stable-diffusion-webui-directml
git submodule init
git submodule update
```

5. Buka file `webui-user.bat`.
6. Isi baris `COMMANDLINE_ARGS` minimal seperti ini:

```bat
set COMMANDLINE_ARGS=--api --use-directml
```

7. Jika VRAM Anda terbatas, tambahkan opsi hemat memori:

```bat
set COMMANDLINE_ARGS=--api --use-directml --opt-sub-quad-attention --lowvram --disable-nan-check
```

8. Simpan file, lalu jalankan `webui-user.bat`.
9. Setelah WebUI terbuka, buka `http://127.0.0.1:7860/docs` untuk memastikan API aktif.
10. Di A1111, buka `Extensions -> Install from URL`, lalu install:

```text
https://github.com/Mikubill/sd-webui-controlnet.git
```

11. Restart penuh A1111.
12. Download model ControlNet `control_v11p_sd15_lineart` lalu letakkan di folder:

```text
stable-diffusion-webui-directml\models\ControlNet
```

13. Jalankan lagi `webui-user.bat`.
14. Setelah itu baru buka `run_app.bat` untuk aplikasi ini.

Catatan penting:

- Anda tetap memerlukan minimal satu checkpoint Stable Diffusion yang kompatibel, biasanya model berbasis SD 1.5, agar A1111 bisa melakukan img2img.
- Jika Anda memakai GPU AMD terintegrasi dengan VRAM kecil, performa bisa sangat lambat atau model gagal dimuat. Dalam kondisi itu, jalankan aplikasi ini tanpa AI atau pindah ke mesin dengan GPU yang lebih kuat.

## AI Cleanup Prompt

The main cleanup step sends `img2img` requests to WebUI using:

- Prompt: `clean vector logo, flat colors, screen printing design, bold outline, minimal colors, vector illustration`
- Negative prompt: `fabric texture, wrinkles, gradient, shadow, blur, noise`
- `denoising_strength=0.35`
- `steps=20`
- `cfg_scale=6`

If WebUI is offline or returns an invalid response, the app falls back to OpenCV cleanup and logs the reason in the GUI console.

## Output

- Main SVG export goes to `output/vector_TIMESTAMP.svg`
- Advanced toolbar features can still export:
  - cutline SVG
  - EPS/PDF through Inkscape
  - batch SVG/EPS/PDF

## Build EXE

Simple build:

```powershell
pip install pyinstaller
pyinstaller --onefile --noconsole app.py --name AutoVectorSablonAI
```

Spec-based build:

```powershell
pyinstaller AutoVectorSablonAI.spec
```

## Build Installer

1. Build the executable first.
2. Open `installer/AutoVectorSablonAI.iss` in Inno Setup.
3. Confirm the `dist/AutoVectorSablonAI/` output exists.
4. Build the installer.

The installer output name is:

`AutoVectorSablonAI_Setup.exe`

## Notes

- The prompt-style workflow is optimized for scanned or photographed t-shirt prints.
- The older advanced pipeline is still present for artwork mode, raster artwork mode, cutline export, and batch processing.
- Processing time depends on hardware and A1111 availability, but the target remains roughly 10-30 seconds per image.
