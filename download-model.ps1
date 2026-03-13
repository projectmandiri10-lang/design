[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$modelFileName = 'RealESRGAN_x4plus.pth'
$modelUrl = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
$minimumExpectedBytes = 1024KB

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$modelsDir = Join-Path $repoRoot 'models'
$destinationPath = Join-Path $modelsDir $modelFileName
$partialPath = "$destinationPath.download"

function Write-Status {
    param(
        [Parameter(Mandatory)]
        [string]$Stage,
        [Parameter(Mandatory)]
        [string]$Message
    )

    Write-Host "[$Stage] $Message"
}

function Fail-Script {
    param(
        [Parameter(Mandatory)]
        [string]$Message
    )

    Write-Status -Stage 'fail' -Message $Message
    exit 1
}

try {
    if (-not (Test-Path -LiteralPath $modelsDir)) {
        New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null
    }

    if ((Test-Path -LiteralPath $destinationPath) -and -not $Force) {
        $existing = Get-Item -LiteralPath $destinationPath
        Write-Status -Stage 'skip' -Message "Model sudah ada di $destinationPath ($([Math]::Round($existing.Length / 1MB, 2)) MB). Gunakan -Force untuk mengunduh ulang."
        exit 0
    }

    if (Test-Path -LiteralPath $partialPath) {
        Remove-Item -LiteralPath $partialPath -Force
    }

    Write-Status -Stage 'download' -Message "Mengunduh $modelFileName dari sumber resmi Real-ESRGAN..."
    Invoke-WebRequest -Uri $modelUrl -OutFile $partialPath -MaximumRedirection 5

    if (-not (Test-Path -LiteralPath $partialPath)) {
        Fail-Script -Message 'File hasil unduhan sementara tidak ditemukan.'
    }

    $downloaded = Get-Item -LiteralPath $partialPath
    if ($downloaded.Length -lt $minimumExpectedBytes) {
        Remove-Item -LiteralPath $partialPath -Force -ErrorAction SilentlyContinue
        Fail-Script -Message "Ukuran file terlalu kecil ($($downloaded.Length) bytes). Unduhan dianggap gagal."
    }

    Move-Item -LiteralPath $partialPath -Destination $destinationPath -Force

    if (-not (Test-Path -LiteralPath $destinationPath)) {
        Fail-Script -Message 'File model tidak ditemukan setelah dipindahkan ke folder tujuan.'
    }

    $finalFile = Get-Item -LiteralPath $destinationPath
    if ($finalFile.Length -lt $minimumExpectedBytes) {
        Remove-Item -LiteralPath $destinationPath -Force -ErrorAction SilentlyContinue
        Fail-Script -Message "Verifikasi gagal karena ukuran file di bawah batas minimum ($minimumExpectedBytes bytes)."
    }

    Write-Status -Stage 'verify' -Message "Model tersimpan di $destinationPath ($([Math]::Round($finalFile.Length / 1MB, 2)) MB)."
    exit 0
}
catch {
    if (Test-Path -LiteralPath $partialPath) {
        Remove-Item -LiteralPath $partialPath -Force -ErrorAction SilentlyContinue
    }

    Fail-Script -Message $_.Exception.Message
}
