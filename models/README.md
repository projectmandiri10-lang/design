# Models

Letakkan file bobot model opsional di folder ini.

- `RealESRGAN_x4plus.pth` - digunakan oleh `ai-engine/vectorizer/upscale.py`

Jika file bobot tidak tersedia, pipeline akan memakai fallback upscaler yang ramah CPU dan mencatat fallback tersebut di metadata respons.
