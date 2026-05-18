Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
& "C:\Users\pinea\anaconda3\Scripts\mkdocs.exe" serve -a 127.0.0.1:8000 *> .mkdocs-serve.log
