# Download and run the Jaeger all-in-one binary (no Docker required).
# Jaeger UI:        http://localhost:16686
# OTLP/HTTP input:  http://localhost:4318
#
# Note: Jaeger 1.x does not publish a standalone all-in-one .exe asset.
# It ships a .zip archive containing jaeger-all-in-one.exe, which we
# download and extract here.

$version = "1.57.0"
$archive = "jaeger-$version-windows-amd64.zip"
$url     = "https://github.com/jaegertracing/jaeger/releases/download/v$version/$archive"
$zip     = "$PSScriptRoot\$archive"
$extract = "$PSScriptRoot\jaeger-$version-windows-amd64"

# Locate the all-in-one binary, downloading/extracting if needed.
$exe = Get-ChildItem -Path $extract -Filter "jaeger-all-in-one.exe" -Recurse -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty FullName

if (-not $exe) {
    if (-not (Test-Path $zip)) {
        Write-Host "Downloading Jaeger $version (~138 MB)..."
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        # curl.exe (shipped with Windows) follows GitHub's CDN redirects and
        # retries far more reliably than Invoke-WebRequest / WebClient.
        curl.exe -L --fail --retry 5 --retry-all-errors -o $zip $url
        if ($LASTEXITCODE -ne 0) {
            if (Test-Path $zip) { Remove-Item $zip -Force }
            Write-Error "Download failed (curl exit $LASTEXITCODE)."
            exit 1
        }
        Write-Host "Downloaded to $zip"
    }

    Write-Host "Extracting..."
    Expand-Archive -Path $zip -DestinationPath $PSScriptRoot -Force

    $exe = Get-ChildItem -Path $extract -Filter "jaeger-all-in-one.exe" -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
    if (-not $exe) {
        Write-Error "jaeger-all-in-one.exe not found in archive after extraction."
        exit 1
    }
}

Write-Host "Starting Jaeger (UI at http://localhost:16686)..."
& $exe `
    --collector.otlp.enabled=true `
    --collector.otlp.http.host-port=":4318"
