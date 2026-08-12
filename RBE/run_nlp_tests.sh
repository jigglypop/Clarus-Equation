#!/bin/bash

echo "=== RBE NLP 레이어 테스트 실행 ==="
echo

# 색상 정의
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 테스트 실행 함수
run_test() {
    local test_name=$1
    echo "테스트 실행: $test_name"
    if cargo test $test_name -- --nocapture; then
        echo -e "${GREEN}✓ $test_name 테스트 통과${NC}"
    else
        echo -e "${RED}✗ $test_name 테스트 실패${NC}"
        exit 1
    fi
    echo
}

# 각 레이어별 테스트 실행
echo "1. RBEEmbedding 테스트"
run_test "embedding::__tests__"

echo "2. RBELayerNorm 테스트"
run_test "layernorm::__tests__"

echo "3. RBEFFN 테스트"
run_test "ffn::__tests__"

echo "4. RBEAttention 테스트"
run_test "attention::__tests__"

echo "=== 모든 NLP 레이어 테스트 통과! ==="
echo

# 예제 실행
echo "예제 프로그램 실행:"
cargo run --example nlp_layers_demo

echo
echo "완료!" 