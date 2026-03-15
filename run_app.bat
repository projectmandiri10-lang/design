@echo off
setlocal

set "ROOT=%~dp0"
pushd "%ROOT%"

set "DEFAULT_PYTHON_EXE=%ROOT%\.venv\Scripts\python.exe"
set "PYTHON_EXE=%DEFAULT_PYTHON_EXE%"
set "USING_CUSTOM_PYTHON=0"
if not "%~1"=="" (
  set "PYTHON_EXE=%~1"
  set "USING_CUSTOM_PYTHON=1"
)

if "%USING_CUSTOM_PYTHON%"=="0" if not exist "%PYTHON_EXE%" (
  echo Virtual environment belum siap. Menjalankan setup otomatis...
  call "%ROOT%setup.bat" --no-pause
  if errorlevel 1 goto :setup_failed
)

if "%USING_CUSTOM_PYTHON%"=="0" set "PYTHON_EXE=%DEFAULT_PYTHON_EXE%"

if not exist "%PYTHON_EXE%" (
  echo Python executable tidak ditemukan: %PYTHON_EXE%
  goto :fail
)

"%PYTHON_EXE%" -c "import requests, PySide6, cv2, PIL" >nul 2>&1
if errorlevel 1 (
  if "%USING_CUSTOM_PYTHON%"=="1" (
    echo Dependency Python belum lengkap pada interpreter ini:
    echo %PYTHON_EXE%
    echo Install dependency secara manual atau jalankan run_app.bat tanpa argumen.
    goto :fail
  )
  echo Dependency Python belum lengkap. Menjalankan setup otomatis...
  call "%ROOT%setup.bat" --no-pause
  if errorlevel 1 goto :setup_failed
)

if "%USING_CUSTOM_PYTHON%"=="0" set "PYTHON_EXE=%DEFAULT_PYTHON_EXE%"

set "POTRACE_CANDIDATE=%ROOT%assets\tools\potrace\potrace-1.16.win64\potrace.exe"
if exist "%POTRACE_CANDIDATE%" (
  set "POTRACE_BIN=%POTRACE_CANDIDATE%"
)

if exist "C:\Program Files\Inkscape\bin\inkscape.exe" (
  set "INKSCAPE_BIN=C:\Program Files\Inkscape\bin\inkscape.exe"
)

if not exist "%ROOT%output" mkdir "%ROOT%output"
if not exist "%ROOT%temp" mkdir "%ROOT%temp"

powershell -NoProfile -Command "exit ([int](-not (Test-NetConnection 127.0.0.1 -Port 7860 -WarningAction SilentlyContinue).TcpTestSucceeded))" >nul 2>&1
if errorlevel 1 (
  echo.
  echo Catatan: AUTOMATIC1111 belum terdeteksi di http://127.0.0.1:7860
  echo Aplikasi tetap bisa dibuka, tetapi AI Cleanup dan ControlNet akan fallback ke mode lokal.
  echo.
)

"%PYTHON_EXE%" "%ROOT%app.py"
set "EXIT_CODE=%ERRORLEVEL%"

popd
exit /b %EXIT_CODE%

:setup_failed
echo Setup otomatis gagal.
echo Jalankan setup.bat secara manual, lalu buka run_app.bat lagi.
goto :fail

:fail
popd
pause
exit /b 1
