#!/usr/bin/env bash
# scripts/create_domain_branches.sh
#
# main 으로부터 5 개 도메인 브랜치를 생성하고, 각 브랜치에서 비-도메인
# 파일을 git rm 한 뒤 단일 커밋으로 정리한다.
#
# 본 스크립트는 한 번만 실행한다 (또는 brain rest 후 재실행 시 기존 브랜치
# 를 강제 삭제하고 다시 만든다).
#
# 사용법:
#   bash scripts/create_domain_branches.sh           # 안전: 이미 있으면 스킵
#   bash scripts/create_domain_branches.sh --force   # 기존 도메인 브랜치 삭제 후 재생성
#
# 전제:
#   - 현재 main 에 있고 working tree 깨끗
#   - python 으로 scripts/branch_split.py 가 실행 가능

set -euo pipefail

DOMAINS=(
    "cosmology"
    "particle"
    "brain-agi"
    "engineering"
    "math-derivations"
)

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERR: working tree dirty. commit or stash first." >&2
    exit 1
fi

CURRENT=$(git rev-parse --abbrev-ref HEAD)
if [[ "$CURRENT" != "main" ]]; then
    echo "ERR: must be on main (current: $CURRENT)" >&2
    exit 1
fi

PY=".venv/Scripts/python.exe"
if [[ ! -x "$PY" ]]; then
    PY="python"
fi

for d in "${DOMAINS[@]}"; do
    BR="domain/${d}"
    if git show-ref --verify --quiet "refs/heads/${BR}"; then
        if [[ "$FORCE" == "1" ]]; then
            echo "[force] delete existing $BR"
            git branch -D "$BR" >/dev/null
        else
            echo "[skip ] $BR already exists (use --force to recreate)"
            continue
        fi
    fi

    echo "[make ] $BR  from main"
    git checkout -b "$BR" main >/dev/null

    REMOVE_FILE=$(mktemp)
    "$PY" scripts/branch_split.py --domain "$d" --print-remove > "$REMOVE_FILE"
    REMOVE_COUNT=$(wc -l < "$REMOVE_FILE" | tr -d ' ')

    if [[ "$REMOVE_COUNT" == "0" ]]; then
        echo "       nothing to remove"
        rm -f "$REMOVE_FILE"
        git checkout main >/dev/null
        continue
    fi

    echo "       git rm $REMOVE_COUNT files"
    # null-terminated 로 안전하게
    tr '\n' '\0' < "$REMOVE_FILE" | xargs -0 git rm -q
    rm -f "$REMOVE_FILE"

    KEEP_COUNT=$(git ls-files | wc -l | tr -d ' ')
    git commit -q -m "branch: prune to ${d} domain (kept ${KEEP_COUNT} files)"
    echo "       kept $KEEP_COUNT files"
    git checkout main >/dev/null
done

echo
echo "== created branches =="
git branch | grep "domain/" || true
