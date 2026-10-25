#Requires -Version 5.1
[CmdletBinding()]
param(
    [string[]] $OutputRoot = @(),
    [string[]] $TargetDir = @(),
    [switch] $Remove,
    [switch] $WhatIf
)

$ErrorActionPreference = 'Stop'

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Add-TrailingSlash([string] $Path) {
    $full = [System.IO.Path]::GetFullPath($Path)
    if ($full.Length -gt 3) {
        return $full.TrimEnd([char[]]@('\', '/'))
    }
    return $full
}

function Get-ExclusionPaths {
    $preference = Get-MpPreference
    if ($null -eq $preference.ExclusionPath) {
        return @()
    }
    return @($preference.ExclusionPath)
}

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$paths = [System.Collections.Generic.List[string]]::new()

$workspaceTarget = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $repo 'target' }
$paths.Add($workspaceTarget)

$tempTarget = Join-Path ([System.IO.Path]::GetTempPath()) 'kirikae-build'
$paths.Add($tempTarget)

foreach ($dir in $TargetDir) {
    $paths.Add($dir)
}
foreach ($root in $OutputRoot) {
    $paths.Add($root)
}

$unique = [System.Collections.Generic.List[string]]::new()
$seen = @{}
foreach ($path in $paths) {
    if ([string]::IsNullOrWhiteSpace($path)) { continue }
    $full = [System.IO.Path]::GetFullPath($path)
    $full = Add-TrailingSlash $full
    $key = $full.ToLowerInvariant()
    if (-not $seen.ContainsKey($key)) {
        $seen[$key] = $true
        $unique.Add($full)
    }
}

if (-not (Get-Command Get-MpPreference -ErrorAction SilentlyContinue)) {
    Write-Error 'Get-MpPreference is unavailable: Windows Defender is not the active antivirus.'
    exit 3
}

if (-not (Test-Administrator)) {
    Write-Host 'This script needs an elevated PowerShell (Administrator). Nothing was changed.'
    Write-Host ''
    Write-Host 'Run this as administrator:'
    foreach ($path in $unique) {
        $verb = if ($Remove) { 'Remove-MpPreference' } else { 'Add-MpPreference' }
        Write-Host ("  {0} -ExclusionPath '{1}'" -f $verb, $path)
    }
    exit 2
}

$existing = Get-ExclusionPaths
$existingKeys = @{}
foreach ($path in $existing) {
    $existingKeys[(Add-TrailingSlash $path).ToLowerInvariant()] = $true
}

$failed = $false
foreach ($path in $unique) {
    $key = $path.ToLowerInvariant()
    if ($Remove) {
        if (-not $existingKeys.ContainsKey($key)) {
            Write-Host "skip (not excluded): $path"
            continue
        }
        $command = "Remove-MpPreference -ExclusionPath '$path'"
        Write-Host $command
        if (-not $WhatIf) {
            try {
                Remove-MpPreference -ExclusionPath $path
            } catch {
                Write-Warning "failed: $command`n$($_.Exception.Message)"
                $failed = $true
            }
        }
        continue
    }

    if ($existingKeys.ContainsKey($key)) {
        Write-Host "already excluded: $path"
        continue
    }
    if (-not (Test-Path -LiteralPath $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
        Write-Host "created: $path"
    }
    $command = "Add-MpPreference -ExclusionPath '$path'"
    Write-Host $command
    if (-not $WhatIf) {
        try {
            Add-MpPreference -ExclusionPath $path
        } catch {
            Write-Warning "failed: $command`n$($_.Exception.Message)"
            $failed = $true
        }
    }
}

if ($failed) { exit 1 }
exit 0
