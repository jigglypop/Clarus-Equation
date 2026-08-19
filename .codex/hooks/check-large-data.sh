#!/bin/bash
# PreToolUse gate: git commit/push 시 연구 데이터 페이로드 업로드를 차단한다.
# stdin으로 hook JSON을 받고, 위반 시 exit 2 + stderr 사유로 도구 호출을 막는다.
# 정책:
#   1) 95MB 초과 파일은 어디서든 차단 (GitHub 하드 리밋 100MB)
#   2) _workspace/ 아래 데이터 확장자(zip|mat|pkl|pickle|npy|npz|h5|hdf5|pt|onnx|parquet|bin|exe)는 크기 무관 차단
# 데이터 원본은 로컬 보관, DOI로 재취득한다 (.gitignore 참조).

INPUT=$(cat)
case "$INPUT" in
  *"git commit"*|*"git push"*) ;;
  *) exit 0 ;;
esac

cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

LIMIT=95000000
DATA_EXT='\.(zip|mat|pkl|pickle|npy|npz|h5|hdf5|pt|onnx|parquet|bin|exe)$'
VIOLATIONS=""

# --- 커밋 전 검사: 스테이징된 파일 ---
if printf '%s' "$INPUT" | grep -q "git commit"; then
  while IFS= read -r -d '' f; do
    [ -f "$f" ] || continue
    if printf '%s' "$f" | grep -qiE "^_workspace/.*$DATA_EXT"; then
      VIOLATIONS="$VIOLATIONS
  - [data-ext] $f"
      continue
    fi
    size=$(wc -c <"$f" 2>/dev/null | tr -d ' ')
    if [ "${size:-0}" -gt "$LIMIT" ]; then
      VIOLATIONS="$VIOLATIONS
  - [>95MB] $f ($((size / 1048576))MB)"
    fi
  done < <(git diff --cached --name-only --diff-filter=AM -z 2>/dev/null)
fi

# --- 푸시 전 검사: 업스트림에 없는 커밋들의 blob ---
if printf '%s' "$INPUT" | grep -q "git push"; then
  range=$(git rev-parse --abbrev-ref '@{u}' 2>/dev/null)
  [ -n "$range" ] && range="$range..HEAD" || range=""
  if [ -n "$range" ]; then
    while IFS= read -r line; do
      sha=${line%% *}
      path=${line#* }
      [ "$sha" = "$path" ] && continue
      if printf '%s' "$path" | grep -qiE "^_workspace/.*$DATA_EXT"; then
        VIOLATIONS="$VIOLATIONS
  - [push:data-ext] $path"
        continue
      fi
      size=$(git cat-file -s "$sha" 2>/dev/null)
      if [ "${size:-0}" -gt "$LIMIT" ]; then
        VIOLATIONS="$VIOLATIONS
  - [push:>95MB] $path ($((size / 1048576))MB)"
      fi
    done < <(git rev-list --objects "$range" 2>/dev/null | grep -iE "$DATA_EXT")
  fi
fi

if [ -n "$VIOLATIONS" ]; then
  {
    echo "BLOCKED: 연구 데이터 페이로드가 커밋/푸시에 포함되어 있습니다."
    echo "$VIOLATIONS"
    echo ""
    echo "조치: git rm --cached <file> 로 언트래킹하고 .gitignore 패턴을 확인하세요."
    echo "데이터 원본은 로컬에 두고 DOI에서 재취득합니다. 95MB 초과는 GitHub가 거부합니다."
  } >&2
  exit 2
fi

exit 0
