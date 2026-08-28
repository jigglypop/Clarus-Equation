# CE_RUN workspace orchestration is retired. Canonical results belong in paper/.
if ($args.Count -gt 0 -and $args[0] -eq 'hook') { exit 0 }
Write-Error 'CE run workspace commands are retired; update canonical paper/ files directly.'
exit 2
