param(
    [switch]$SkipSemantic
)

# This wrapper checks document syntax, a bounded canonical-role contract, and
# arithmetic recomputation.  It is not a proof of G4 physics or G5 likelihoods.
$ErrorActionPreference = 'Stop'
$DocsRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$MarkdownFiles = Get-ChildItem -LiteralPath $DocsRoot -Recurse -File -Filter '*.md'
$Failures = [System.Collections.Generic.List[string]]::new()

function Add-Failure([string]$Kind, [string]$Path, [string]$Detail) {
    $rootUri = [System.Uri]::new(($DocsRoot.TrimEnd('\') + '\'))
    $pathUri = [System.Uri]::new($Path)
    $relative = [System.Uri]::UnescapeDataString(
        $rootUri.MakeRelativeUri($pathUri).ToString()
    ).Replace('/', '\')
    $Failures.Add("[$Kind] ${relative}: $Detail")
}

function Test-IsEscaped([string]$Line, [int]$Index) {
    $slashCount = 0
    for ($cursor = $Index - 1; $cursor -ge 0 -and $Line[$cursor] -eq '\'; $cursor--) {
        $slashCount++
    }
    return (($slashCount % 2) -eq 1)
}

function Get-UnescapedCharacterCount([string]$Line, [char]$Character) {
    $count = 0
    for ($index = 0; $index -lt $Line.Length; $index++) {
        if ($Line[$index] -eq $Character -and -not (Test-IsEscaped $Line $index)) {
            $count++
        }
    }
    return $count
}

function Get-MarkdownTableColumnCount([string]$Line) {
    $trimmed = $Line.Trim()
    if ([string]::IsNullOrWhiteSpace($trimmed)) {
        return 0
    }

    $pipeCount = Get-UnescapedCharacterCount $trimmed '|'
    if ($pipeCount -eq 0) {
        return 0
    }

    $leadingPipe = if ($trimmed[0] -eq '|') { 1 } else { 0 }
    $lastIndex = $trimmed.Length - 1
    $trailingPipe = if (
        $trimmed[$lastIndex] -eq '|' -and
        -not (Test-IsEscaped $trimmed $lastIndex)
    ) { 1 } else { 0 }

    return ($pipeCount + 1 - $leadingPipe - $trailingPipe)
}

function Test-IsMarkdownTableSeparator([string]$Line) {
    $trimmed = $Line.Trim()
    if ([string]::IsNullOrWhiteSpace($trimmed) -or
        (Get-UnescapedCharacterCount $trimmed '|') -eq 0) {
        return $false
    }

    if ($trimmed.StartsWith('|')) {
        $trimmed = $trimmed.Substring(1)
    }
    $lastIndex = $trimmed.Length - 1
    if ($lastIndex -ge 0 -and
        $trimmed[$lastIndex] -eq '|' -and
        -not (Test-IsEscaped $trimmed $lastIndex)) {
        $trimmed = $trimmed.Substring(0, $lastIndex)
    }

    $cells = $trimmed -split '\|'
    if ($cells.Count -lt 2) {
        return $false
    }
    foreach ($cell in $cells) {
        if ($cell -notmatch '^\s*:?-{3,}:?\s*$') {
            return $false
        }
    }
    return $true
}

$codeFenceToken = -join (([char]96).ToString() * 3)

foreach ($file in $MarkdownFiles) {
    $text = Get-Content -LiteralPath $file.FullName -Encoding utf8 -Raw
    $prose = [regex]::Replace($text, '(?ms)^\s*```.*?^\s*```\s*$', '')

    $controlCharacter = [regex]::Match(
        $text,
        '[\x00-\x08\x0B\x0C\x0E-\x1F]'
    )
    if ($controlCharacter.Success) {
        Add-Failure 'CONTROL_CHAR' $file.FullName (
            'contains a forbidden C0 control character'
        )
    }

    $inlineParenOpen = [regex]::Matches($prose, '\\\(').Count
    $inlineParenClose = [regex]::Matches($prose, '\\\)').Count
    if ($inlineParenOpen -ne $inlineParenClose) {
        Add-Failure 'INLINE_MATH' $file.FullName (
            "mismatched \( and \): $inlineParenOpen vs $inlineParenClose"
        )
    }

    $displayBracketOpen = [regex]::Matches($prose, '\\\[').Count
    $displayBracketClose = [regex]::Matches($prose, '\\\]').Count
    if ($displayBracketOpen -ne $displayBracketClose) {
        Add-Failure 'DISPLAY_MATH' $file.FullName (
            "mismatched \[ and \]: $displayBracketOpen vs $displayBracketClose"
        )
    }

    $lines = [regex]::Split($text, '\r?\n')
    $insideCodeFence = $false
    for ($lineIndex = 0; $lineIndex -lt $lines.Count; $lineIndex++) {
        $line = $lines[$lineIndex]
        if ($line -match ('^\s*' + [regex]::Escape($codeFenceToken))) {
            $insideCodeFence = -not $insideCodeFence
            continue
        }
        if ($insideCodeFence) {
            continue
        }

        $dollarCount = Get-UnescapedCharacterCount $line '$'
        if (($dollarCount % 2) -ne 0) {
            Add-Failure 'INLINE_DOLLAR' $file.FullName (
                "line $($lineIndex + 1) has odd unescaped dollar count $dollarCount"
            )
        }

        if (-not (Test-IsMarkdownTableSeparator $line)) {
            continue
        }

        $expectedColumns = Get-MarkdownTableColumnCount $line
        if ($lineIndex -eq 0) {
            Add-Failure 'TABLE' $file.FullName (
                "line $($lineIndex + 1) has no header row"
            )
            continue
        }

        $headerColumns = Get-MarkdownTableColumnCount $lines[$lineIndex - 1]
        if ($headerColumns -ne $expectedColumns) {
            Add-Failure 'TABLE' $file.FullName (
                "line $lineIndex header has $headerColumns columns; separator has $expectedColumns"
            )
        }

        for ($bodyIndex = $lineIndex + 1; $bodyIndex -lt $lines.Count; $bodyIndex++) {
            $bodyLine = $lines[$bodyIndex]
            if ([string]::IsNullOrWhiteSpace($bodyLine)) {
                break
            }
            $bodyColumns = Get-MarkdownTableColumnCount $bodyLine
            if ($bodyColumns -eq 0) {
                break
            }
            if ($bodyColumns -ne $expectedColumns) {
                Add-Failure 'TABLE' $file.FullName (
                    "line $($bodyIndex + 1) has $bodyColumns columns; expected $expectedColumns"
                )
            }
        }
    }

    $h1Count = [regex]::Matches($prose, '(?m)^#\s+[^#\r\n]').Count
    if ($h1Count -ne 1) {
        Add-Failure 'H1' $file.FullName "expected 1, found $h1Count"
    }

    $displayMathFenceCount = [regex]::Matches($prose, '(?m)^\s*\$\$\s*$').Count
    if (($displayMathFenceCount % 2) -ne 0) {
        Add-Failure 'MATH_FENCE' $file.FullName "odd dollar-fence count $displayMathFenceCount"
    }

    $codeFenceCount = [regex]::Matches($text, '(?m)^\s*```').Count
    if (($codeFenceCount % 2) -ne 0) {
        Add-Failure 'CODE_FENCE' $file.FullName "odd code-fence count $codeFenceCount"
    }

    $linkMatches = [regex]::Matches($prose, '(?<!!)\[[^\]]+\]\((?<target>[^\)]+)\)')
    foreach ($match in $linkMatches) {
        $target = $match.Groups['target'].Value.Trim()
        if ($target.StartsWith('<') -and $target.EndsWith('>')) {
            $target = $target.Substring(1, $target.Length - 2)
        }
        $target = ($target -split '\s+"', 2)[0]
        $target = ($target -split '#', 2)[0]
        if ([string]::IsNullOrWhiteSpace($target) -or
            $target -match '^(https?://|mailto:|app://)') {
            continue
        }
        if ($target -match '[\{\}]' -or $target.StartsWith('\')) {
            continue
        }
        if ($target -notmatch '[/\\]' -and $target -notmatch '\.[A-Za-z0-9]+$') {
            continue
        }
        $decoded = [System.Uri]::UnescapeDataString($target)
        $candidate = Join-Path $file.DirectoryName $decoded
        if (-not (Test-Path -LiteralPath $candidate)) {
            Add-Failure 'LINK' $file.FullName "missing target $target"
        }
    }
}

$numericScript = Join-Path $PSScriptRoot 'verify_numeric_consistency.py'
$documentScript = Join-Path $PSScriptRoot 'verify_document_contract.py'
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $pythonCommand) {
    Add-Failure 'DOCUMENT_CONTRACT' $documentScript 'python command was not found'
    Add-Failure 'NUMERIC' $numericScript 'python command was not found'
}
else {
    $documentArguments = @($documentScript)
    if ($SkipSemantic) {
        $documentArguments += '--syntax-only'
    }
    $documentOutput = & $pythonCommand.Source @documentArguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        $documentOutput | ForEach-Object { Write-Host $_ }
        Add-Failure 'DOCUMENT_CONTRACT' $documentScript (
            'document verifier reported failures above'
        )
    }
    else {
        $documentOutput | ForEach-Object { Write-Host $_ }
    }

    $numericOutput = & $pythonCommand.Source $numericScript 2>&1
    if ($LASTEXITCODE -ne 0) {
        Add-Failure 'NUMERIC' $numericScript ($numericOutput -join ' | ')
    }
    else {
        $numericOutput | ForEach-Object { Write-Host $_ }
    }
}

if ($Failures.Count -gt 0) {
    Write-Host "DOCUMENT/SYNTAX + CANONICAL ARITHMETIC GATE: FAIL"
    $Failures | Sort-Object | ForEach-Object { Write-Host $_ }
    if ($SkipSemantic) {
        Write-Host 'Scope: Markdown syntax and canonical arithmetic recomputation; bounded semantic-contract checks were skipped.'
    }
    else {
        Write-Host 'Scope: Markdown syntax, bounded semantic contracts, and canonical arithmetic recomputation; this is not a full physics or likelihood proof.'
    }
    exit 1
}

Write-Host "DOCUMENT/SYNTAX + CANONICAL ARITHMETIC GATE: PASS"
Write-Host "Markdown files checked: $($MarkdownFiles.Count)"
if ($SkipSemantic) {
    Write-Host 'Scope: Markdown syntax and canonical arithmetic recomputation; bounded semantic-contract checks were skipped.'
}
else {
    Write-Host 'Scope: Markdown syntax, bounded semantic contracts, and canonical arithmetic recomputation; this is not a full physics or likelihood proof.'
}
exit 0
