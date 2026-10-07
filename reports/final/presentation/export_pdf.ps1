param()
$ErrorActionPreference = 'Stop'

# Local export only; the browser profile and logs stay beside this source.
$presentationRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$presentationEdge = Join-Path ${env:ProgramFiles(x86)} 'Microsoft\Edge\Application\msedge.exe'
if (-not (Test-Path -LiteralPath $presentationEdge)) {
    $presentationEdge = Join-Path $env:ProgramFiles 'Microsoft\Edge\Application\msedge.exe'
}
if (-not (Test-Path -LiteralPath $presentationEdge)) {
    throw 'Microsoft Edge is unavailable; no packages will be installed.'
}
$presentationSource = Join-Path $presentationRoot 'presentation.html'
$presentationPdf = Join-Path $presentationRoot 'presentation.pdf'
$presentationTemporary = [IO.Path]::GetFullPath((Join-Path $presentationRoot '.export'))
if (-not $presentationTemporary.StartsWith($presentationRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'The export directory is outside the presentation directory.'
}
if (Test-Path -LiteralPath $presentationTemporary) {
    throw 'The temporary export directory already exists; inspect it before another export.'
}
New-Item -ItemType Directory -Path $presentationTemporary | Out-Null
$presentationProfile = Join-Path $presentationTemporary 'profile'
$presentationUri = ([Uri]$presentationSource).AbsoluteUri
$presentationArguments = @(
    '--headless', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
    '--disable-extensions', '--disable-background-networking', '--disable-component-update',
    '--disable-sync', '--metrics-recording-only', '--no-pdf-header-footer',
    '--disable-breakpad', '--disable-crash-reporter', '--no-sandbox',
    '--disable-features=RendererCodeIntegrity', '--proxy-server=127.0.0.1:9',
    '--run-all-compositor-stages-before-draw', '--virtual-time-budget=2500',
    ('--user-data-dir="{0}"' -f $presentationProfile),
    ('--print-to-pdf="{0}"' -f $presentationPdf),
    ('"{0}"' -f $presentationUri)
)
try {
    $presentationProcess = Start-Process -FilePath $presentationEdge -ArgumentList $presentationArguments -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput (Join-Path $presentationTemporary 'stdout.log') -RedirectStandardError (Join-Path $presentationTemporary 'stderr.log')
    if ($presentationProcess.ExitCode -ne 0) {
        Get-Content -LiteralPath (Join-Path $presentationTemporary 'stderr.log') -Encoding UTF8
        throw ('Edge exit code: {0}' -f $presentationProcess.ExitCode)
    }
    if (-not (Test-Path -LiteralPath $presentationPdf)) { throw 'Edge did not create a PDF.' }
    if ((Get-Item -LiteralPath $presentationPdf).Length -lt 10000) { throw 'The PDF is unexpectedly small.' }
    Write-Output ('Created presentation.pdf ({0} bytes).' -f (Get-Item -LiteralPath $presentationPdf).Length)
}
finally {
    # Remove only the verified temporary directory created by this invocation.
    if (Test-Path -LiteralPath $presentationTemporary) {
        Remove-Item -LiteralPath $presentationTemporary -Recurse -Force
    }
}
