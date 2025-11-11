# codex_proxy.ps1
$env:HTTP_PROXY = "http://127.0.0.1:7897"
$env:HTTPS_PROXY = $env:HTTP_PROXY
& "$env:AppData\npm\codex.cmd" @args
