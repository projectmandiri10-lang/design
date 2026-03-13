@echo off
setlocal

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

title AI Screen Printing Vectorizer Launcher

echo [INFO] Menjalankan AI Screen Printing Vectorizer...
echo.

where npm >nul 2>nul
if errorlevel 1 (
  echo [ERROR] npm tidak ditemukan.
  echo [ERROR] Install Node.js LTS terlebih dahulu, lalu coba lagi.
  echo.
  pause
  exit /b 1
)

if not exist "package.json" (
  echo [ERROR] File package.json root tidak ditemukan.
  echo [ERROR] Pastikan file ini dijalankan dari folder project yang benar.
  echo.
  pause
  exit /b 1
)

if not exist "backend\package.json" (
  echo [ERROR] File backend\package.json tidak ditemukan.
  echo.
  pause
  exit /b 1
)

if not exist "frontend\package.json" (
  echo [ERROR] File frontend\package.json tidak ditemukan.
  echo.
  pause
  exit /b 1
)

if not exist "node_modules" (
  echo [ERROR] Dependensi JavaScript belum terpasang.
  echo [ERROR] Jalankan ^`npm install^` dari root project terlebih dahulu.
  echo.
  pause
  exit /b 1
)

echo [INFO] Membuka terminal backend...
start "AI Vectorizer Backend" cmd /k "cd /d ""%ROOT_DIR%"" && npm run dev --workspace backend"

timeout /t 2 /nobreak >nul

echo [INFO] Membuka terminal frontend...
start "AI Vectorizer Frontend" cmd /k "cd /d ""%ROOT_DIR%"" && npm run dev --workspace frontend"

echo.
echo [INFO] Backend dan frontend sudah diminta untuk dijalankan.
echo [INFO] Jika backend gagal, cek Python 3.11, Potrace, dan dependensi AI engine.
echo [INFO] Jika frontend gagal, cek hasil ^`npm install^`.
echo.
echo Backend  : http://localhost:3001
echo Frontend : http://localhost:5173
echo.
pause
exit /b 0
