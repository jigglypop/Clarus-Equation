#!/bin/bash

echo "🚀 RBE 한국어 sLLM 데모 실행"
echo "=========================="

# 1. 모델 다운로드 확인
if [ ! -d "models/kominilm-23m" ]; then
    echo "📥 한국어 모델 다운로드 중..."
    python setup_korean_test_model.py
    if [ $? -ne 0 ]; then
        echo "❌ 모델 다운로드 실패!"
        exit 1
    fi
else
    echo "✅ 모델이 이미 다운로드되어 있습니다."
fi

# 2. 빌드
echo ""
echo "🔨 프로젝트 빌드 중..."
cargo build --release --example korean_sllm_demo
if [ $? -ne 0 ]; then
    echo "❌ 빌드 실패!"
    exit 1
fi

# 3. 실행
echo ""
echo "🚀 데모 실행 중..."
echo "=================="
cargo run --release --example korean_sllm_demo

echo ""
echo "✨ 완료!" 