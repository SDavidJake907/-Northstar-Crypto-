param(
    [string]$Root = ".",
    [string]$OutFile = ".\blob_manifest.sha256"
)

$ErrorActionPreference = "Stop"

$rootPath = (Resolve-Path $Root).Path
$outPath = if ([System.IO.Path]::IsPathRooted($OutFile)) {
    $OutFile
} else {
    Join-Path $rootPath $OutFile
}

Get-ChildItem -Path $rootPath -Recurse -File |
    Sort-Object FullName |
    ForEach-Object {
        $hash = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
        $rel = Resolve-Path -Relative $_.FullName
        "{0} {1}" -f $hash, $rel
    } |
    Out-File -FilePath $outPath -Encoding ascii

Write-Host "Manifest written: $outPath"
