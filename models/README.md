# ONNX Models

Current models in this folder are real ONNX models (not placeholders):

- `sablon_detector.onnx` -> U2NetP salient segmentation (detector stage)
- `realesrgan_x4.onnx` -> Real-ESRGAN x4 ONNX (restoration stage)
- `fabric_unet_denoise.onnx` -> U2NetP ONNX (fabric texture guidance stage)

Download sources used:

- `https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx`
- `https://huggingface.co/AXERA-TECH/Real-ESRGAN/resolve/main/onnx/realesrgan-x4.onnx?download=true`

Fallback behavior:

- If any model is missing/corrupt, the pipeline falls back to classical OpenCV processing and continues without crashing.
