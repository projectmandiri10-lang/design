# Models

Letakkan file bobot model lokal di folder ini.

- `RealESRGAN_x4plus.pth` - digunakan oleh `ai-engine/vectorizer/upscale.py`

Cara utama untuk mengisi folder ini:

- jalankan `.\download-model.ps1` dari root repo
- file akan disimpan sebagai `models/RealESRGAN_x4plus.pth`

Catatan:

- file bobot tidak ikut Git karena sudah diabaikan di `.gitignore`
- jika file bobot tidak tersedia, pipeline akan memakai fallback upscaler yang ramah CPU dan mencatat fallback tersebut di metadata respons
