#!/usr/bin/env python3
"""
GPT-2 토크나이저 다운로드 스크립트
"""

import os
import json
from transformers import GPT2Tokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

def download_and_convert_tokenizer():
    """HuggingFace에서 GPT-2 토크나이저를 다운로드하고 tokenizers 형식으로 변환"""
    
    print("🔽 GPT-2 토크나이저 다운로드 중...")
    
    # 디렉토리 생성
    os.makedirs("models", exist_ok=True)
    
    # HuggingFace에서 토크나이저 다운로드
    tokenizer_hf = GPT2Tokenizer.from_pretrained("gpt2")
    
    # 토크나이저 파일 저장
    tokenizer_hf.save_pretrained("models/gpt2_tokenizer_hf")
    
    # tokenizers 라이브러리 형식으로 변환
    print("🔄 토크나이저 형식 변환 중...")
    
    # 어휘 파일 로드
    with open("models/gpt2_tokenizer_hf/vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    # 병합 파일 로드 - 튜플 리스트로 변환
    merges_list = []
    with open("models/gpt2_tokenizer_hf/merges.txt", "r", encoding="utf-8") as f:
        lines = f.read().split("\n")[1:]  # 첫 줄은 버전 정보
        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) == 2:
                    merges_list.append((parts[0], parts[1]))
    
    # BPE 토크나이저 생성
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges_list))
    
    # Pre-tokenizer 설정 (GPT-2와 동일하게)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Decoder 설정
    tokenizer.decoder = decoders.ByteLevel()
    
    # Post-processor 설정 (특별 토큰 처리)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # 특별 토큰 추가
    tokenizer.add_special_tokens(["<|endoftext|>"])
    
    # 저장
    tokenizer.save("models/tokenizer.json")
    
    print("✅ 토크나이저 저장 완료: models/tokenizer.json")
    
    # 테스트
    print("\n🧪 토크나이저 테스트:")
    test_text = "Hello, world! This is a test."
    encoding = tokenizer.encode(test_text)
    print(f"  원본: {test_text}")
    print(f"  토큰 ID: {encoding.ids}")
    print(f"  토큰: {encoding.tokens}")
    
    decoded = tokenizer.decode(encoding.ids)
    print(f"  복원: {decoded}")
    
    # 한국어 테스트
    test_korean = "안녕하세요, 세계!"
    encoding_kr = tokenizer.encode(test_korean)
    print(f"\n  한국어: {test_korean}")
    print(f"  토큰 ID: {encoding_kr.ids[:10]}...")
    print(f"  토큰 수: {len(encoding_kr.ids)}")

if __name__ == "__main__":
    try:
        download_and_convert_tokenizer()
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("transformers 설치가 필요합니다: pip install transformers tokenizers") 