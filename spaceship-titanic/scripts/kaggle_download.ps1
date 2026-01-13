param(
  [string]$OutDir = "$PSScriptRoot\..\data\raw"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
  Write-Error "kaggle CLI not found. Activate .venv and run: pip install kaggle"
  exit 1
}

if (-not (Test-Path $OutDir)) {
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
}

Write-Host "Downloading Kaggle Spaceship Titanic dataset to $OutDir ..."

# Download competition data
kaggle competitions download -c spaceship-titanic -p $OutDir

# Unzip and clean up
Get-ChildItem -Path $OutDir -Filter *.zip | ForEach-Object {
  Expand-Archive -Path $_.FullName -DestinationPath $OutDir -Force
  Remove-Item $_.FullName -Force
}

Write-Host "Done. Files in: $OutDir"
