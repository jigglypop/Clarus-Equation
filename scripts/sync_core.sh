#!/usr/bin/env bash
# scripts/sync_core.sh
#
# main 의 CORE 변경분을 5 개 도메인 브랜치에 전파한다.
# - 모든 도메인 브랜치에 main 을 merge 한다 (fast-forward 가능하면 ff, 아니면 commit)
# - 충돌이 발생한 도메인 브랜치는 보고만 하고 중단 (수동 해결)
# - 도메인 브랜치는 main 의 슈퍼셋이 아니므로 항상 conflict 가능성이 있음
#
# 사용법:
#   bash scripts/sync_core.sh           # 모든 도메인 브랜치 동기화 시도
#   bash scripts/sync_core.sh --check   # merge 가 필요한지 dry-run 만 (체크아웃 변경 없음)
#
# 전제:
#   - 깨끗한 working tree (git status 무변경)
#   - main 브랜치가 origin/main 과 일치 (또는 무관)

set -euo pipefail

DOMAINS=(
    "domain/cosmology"
    "domain/particle"
    "domain/brain-agi"
    "domain/engineering"
    "domain/math-derivations"
)

CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
    CHECK_ONLY=1
fi

# --- 사전 점검 ---
if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERR: working tree dirty. commit or stash first." >&2
    exit 1
fi

START_BRANCH=$(git rev-parse --abbrev-ref HEAD)
MAIN_SHA=$(git rev-parse main)

echo "main is at ${MAIN_SHA:0:8}"
echo "starting from branch: $START_BRANCH"
echo

CONFLICTS=()
SYNCED=()
ALREADY=()

for branch in "${DOMAINS[@]}"; do
    if ! git show-ref --verify --quiet "refs/heads/${branch}"; then
        echo "skip ${branch} (does not exist)"
        continue
    fi

    BRANCH_SHA=$(git rev-parse "$branch")
    AHEAD=$(git rev-list --count "${MAIN_SHA}..${branch}" 2>/dev/null || echo 0)
    BEHIND=$(git rev-list --count "${branch}..${MAIN_SHA}" 2>/dev/null || echo 0)

    if [[ "$BEHIND" == "0" ]]; then
        echo "[ok ] ${branch}  already contains main (ahead=${AHEAD})"
        ALREADY+=("$branch")
        continue
    fi

    echo "[try] ${branch}  ahead=${AHEAD} behind=${BEHIND}"

    if [[ "$CHECK_ONLY" == "1" ]]; then
        continue
    fi

    git checkout "$branch" >/dev/null 2>&1
    if git merge --no-edit main >/dev/null 2>&1; then
        echo "       merged ok"
        SYNCED+=("$branch")
    else
        echo "       CONFLICT - aborting merge for $branch"
        git merge --abort
        CONFLICTS+=("$branch")
    fi
done

git checkout "$START_BRANCH" >/dev/null 2>&1

echo
echo "== summary =="
echo "  already synced : ${#ALREADY[@]}"
echo "  newly merged   : ${#SYNCED[@]}"
echo "  conflicts      : ${#CONFLICTS[@]}"
if (( ${#CONFLICTS[@]} > 0 )); then
    echo
    echo "Resolve manually:"
    for b in "${CONFLICTS[@]}"; do
        echo "  git checkout $b && git merge main   # then resolve and commit"
    done
    exit 2
fi
