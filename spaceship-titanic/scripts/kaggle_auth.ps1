param(
  [string]$KaggleDir = "$env:USERPROFILE\.kaggle"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $KaggleDir)) {
  New-Item -ItemType Directory -Force -Path $KaggleDir | Out-Null
}

$u = Read-Host "Kaggle username"
$kSecure = Read-Host "Kaggle key" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($kSecure)
$k = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)

$payload = "{`n  \"username\": \"$u\",`n  \"key\": \"$k\"`n}`n"
$path = Join-Path $KaggleDir "kaggle.json"
Set-Content -Encoding ascii -Path $path -Value $payload

Write-Host "Wrote: $path"
