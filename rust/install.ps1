param(
    [Parameter(Mandatory = $true)][ValidatePattern('^(0|[1-9][0-9]{0,5})\.(0|[1-9][0-9]{0,5})\.(0|[1-9][0-9]{0,5})$')][string]$Version,
    [string]$Prefix = "$env:LOCALAPPDATA\Programs\ml-exp\bin",
    [switch]$NoPath
)
$ErrorActionPreference = 'Stop'
if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
    throw 'This Windows release supports x86_64. Use the Linux ARM64 client in WSL for an ARM64 host.'
}
if (-not [IO.Path]::IsPathRooted($Prefix)) { throw 'Use an absolute installation directory.' }
New-Item -ItemType Directory -Force -Path $Prefix | Out-Null
if ((Get-Item -LiteralPath $Prefix).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Installation directory must not be a link.' }
$Destination = Join-Path $Prefix 'ml-exp.exe'
if ((Test-Path -LiteralPath $Destination) -and ((Get-Item -LiteralPath $Destination).Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Refusing to replace a linked executable.' }
$Scratch = Join-Path $Prefix ('.ml-exp-install-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $Scratch | Out-Null
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $Base = "https://github.com/NotDesigned/ml-experiment-control/releases/download/native-v$Version"
    $Asset = 'ml-exp-x86_64-pc-windows-msvc.exe'
    $Sums = Join-Path $Scratch 'SHA256SUMS'
    $Candidate = Join-Path $Scratch 'ml-exp.exe'
    Invoke-WebRequest -UseBasicParsing -TimeoutSec 60 -Uri "$Base/SHA256SUMS" -OutFile $Sums
    if ((Get-Item $Sums).Length -gt 65536) { throw 'Checksum document exceeds limit.' }
    $Rows = @(Get-Content -LiteralPath $Sums | Where-Object { $_ -match ('^[0-9a-f]{64}\s+' + [Regex]::Escape($Asset) + '$') })
    if ($Rows.Count -ne 1) { throw 'Release has no unique Windows client checksum.' }
    $Expected = ($Rows[0] -split '\s+')[0]
    Invoke-WebRequest -UseBasicParsing -TimeoutSec 180 -Uri "$Base/$Asset" -OutFile $Candidate
    if ((Get-Item $Candidate).Length -gt 134217728 -or (Get-FileHash -Algorithm SHA256 -LiteralPath $Candidate).Hash.ToLowerInvariant() -ne $Expected) { throw 'Downloaded executable hash differs.' }
    $Actual = (& $Candidate --version | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $Actual -ne "ml-exp $Version") { throw 'Downloaded executable version differs.' }
    Move-Item -Force -LiteralPath $Candidate -Destination $Destination
    if (-not $NoPath) {
        $UserPath = [Environment]::GetEnvironmentVariable('Path', 'User')
        if (($UserPath -split ';') -notcontains $Prefix) {
            [Environment]::SetEnvironmentVariable('Path', (($UserPath.TrimEnd(';') + ';' + $Prefix).TrimStart(';')), 'User')
        }
        if (($env:Path -split ';') -notcontains $Prefix) { $env:Path += ";$Prefix" }
    }
    Write-Output "Installed ml-exp $Version to $Destination"
} finally {
    Remove-Item -Recurse -Force -LiteralPath $Scratch
}
