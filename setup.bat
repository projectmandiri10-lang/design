@echo off
setlocal

set "ROOT=%~dp0"
set "NO_PAUSE=0"
if /I "%~1"=="--no-pause" set "NO_PAUSE=1"

pushd "%ROOT%"

set "EXIT_CODE=0"

echo =============================================
echo AUTO VECTOR SABLON AI - WINDOWS SETUP
echo =============================================
echo.

call :find_python
if errorlevel 1 goto :python_missing

if not exist "%ROOT%\.venv\Scripts\python.exe" (
  echo [1/4] Membuat virtual environment...
  call :run_python -m venv "%ROOT%\.venv"
  if errorlevel 1 goto :venv_failed
) else (
  echo [1/4] Virtual environment sudah ada.
)

set "VENV_PYTHON=%ROOT%\.venv\Scripts\python.exe"

echo [2/4] Update pip, setuptools, wheel...
"%VENV_PYTHON%" -m pip install --upgrade pip setuptools wheel > "%ROOT%pip_install.log" 2>&1
if errorlevel 1 goto :pip_failed

echo [3/4] Install dependency aplikasi...
"%VENV_PYTHON%" -m pip install -r "%ROOT%requirements.txt" >> "%ROOT%pip_install.log" 2>&1
if errorlevel 1 goto :pip_failed

echo [4/4] Menyiapkan folder kerja...
if not exist "%ROOT%output" mkdir "%ROOT%output"
if not exist "%ROOT%temp" mkdir "%ROOT%temp"

echo.
echo Setup selesai.
echo.
echo Jalankan file run_app.bat untuk membuka aplikasi.
echo Jika ingin fitur AI Cleanup dan ControlNet, jalankan juga
echo AUTOMATIC1111 WebUI dengan opsi --api di http://127.0.0.1:7860
goto :finish

:python_missing
set "EXIT_CODE=1"
echo Python 3.10 atau lebih baru tidak ditemukan.
echo Install Python dari https://www.python.org/downloads/
echo Saat install, centang "Add Python to PATH".
goto :finish

:venv_failed
set "EXIT_CODE=1"
echo Gagal membuat virtual environment.
goto :finish

:pip_failed
set "EXIT_CODE=1"
echo Gagal install dependency.
echo Lihat log: "%ROOT%pip_install.log"
goto :finish

:finish
popd
if "%NO_PAUSE%"=="0" pause
exit /b %EXIT_CODE%

:find_python
set "BOOTSTRAP_PYTHON="
set "BOOTSTRAP_ARGS="

call :try_python py -3.11
if defined BOOTSTRAP_PYTHON exit /b 0

call :try_python py -3.10
if defined BOOTSTRAP_PYTHON exit /b 0

call :try_python py -3
if defined BOOTSTRAP_PYTHON exit /b 0

call :try_python py
if defined BOOTSTRAP_PYTHON exit /b 0

call :try_python python
if defined BOOTSTRAP_PYTHON exit /b 0

exit /b 1

:try_python
set "TRY_CMD=%~1"
set "TRY_ARG=%~2"

if defined TRY_ARG (
  "%TRY_CMD%" %TRY_ARG% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
) else (
  "%TRY_CMD%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
)

if errorlevel 1 exit /b 0

set "BOOTSTRAP_PYTHON=%TRY_CMD%"
set "BOOTSTRAP_ARGS=%TRY_ARG%"
exit /b 0

:run_python
if defined BOOTSTRAP_ARGS (
  "%BOOTSTRAP_PYTHON%" %BOOTSTRAP_ARGS% %*
) else (
  "%BOOTSTRAP_PYTHON%" %*
)
exit /b %ERRORLEVEL%
