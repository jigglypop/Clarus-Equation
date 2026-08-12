#!/bin/bash

# RBE GPT-2 추론 실행 스크립트

echo "🚀 RBE GPT-2 추론 실행"
echo "================================"

# 빌드
echo "⏳ 프로젝트 빌드 중..."
cargo build --release --bin rbe_inference

# 기본 실행
echo -e "\n📝 기본 프롬프트로 실행..."
./target/release/rbe_inference \
    --model-dir ./models/rbe_compressed \
    --prompt "The quick brown fox" \
    --max-tokens 50 \
    --temperature 0.8

# 한국어 테스트
echo -e "\n\n📝 한국어 프롬프트로 실행..."
./target/release/rbe_inference \
    --model-dir ./models/rbe_compressed \
    --prompt "인공지능의 미래는" \
    --max-tokens 100 \
    --temperature 0.9 \
    --top-p 0.95

# 창의적 생성 테스트
echo -e "\n\n📝 창의적 생성 테스트..."
./target/release/rbe_inference \
    --model-dir ./models/rbe_compressed \
    --prompt "Once upon a time in a magical kingdom" \
    --max-tokens 150 \
    --temperature 1.2 \
    --top-p 0.98 \
    --repetition-penalty 1.2

echo -e "\n✅ 추론 테스트 완료!" 