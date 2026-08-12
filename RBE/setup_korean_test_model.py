#!/usr/bin/env python3
"""
KoMiniLM-23M 한국어 테스트 모델 설정 스크립트
"""

import os
import json
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from pathlib import Path

def setup_korean_test_model():
    """KoMiniLM-23M 모델을 다운로드하고 설정"""
    
    MODEL_ID = "BM-K/KoMiniLM"
    MODEL_DIR = Path("models/kominilm-23m")
    
    print("🇰🇷 한국어 테스트 모델 설정 시작")
    print(f"📦 모델: {MODEL_ID}")
    print(f"📁 저장 경로: {MODEL_DIR}")
    
    # 디렉토리 생성
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. 설정 다운로드
        print("\n📋 모델 설정 다운로드 중...")
        config = AutoConfig.from_pretrained(MODEL_ID)
        config.save_pretrained(MODEL_DIR)
        
        # 설정 정보 출력
        print(f"  ✅ 모델 타입: {config.model_type}")
        print(f"  ✅ 숨겨진 크기: {config.hidden_size}")
        print(f"  ✅ 레이어 수: {config.num_hidden_layers}")
        print(f"  ✅ 어텐션 헤드: {config.num_attention_heads}")
        print(f"  ✅ 어휘 크기: {config.vocab_size}")
        
        # 2. 토크나이저 다운로드
        print("\n🔤 토크나이저 다운로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.save_pretrained(MODEL_DIR)
        
        # 토크나이저 테스트
        test_korean_text = "안녕하세요! RBE 시스템을 테스트하고 있습니다."
        tokens = tokenizer.encode(test_korean_text)
        decoded = tokenizer.decode(tokens)
        
        print(f"  ✅ 토크나이저 테스트:")
        print(f"    원본: {test_korean_text}")
        print(f"    토큰 수: {len(tokens)}")
        print(f"    복원: {decoded}")
        
        # 3. 모델 다운로드
        print("\n🧠 모델 가중치 다운로드 중...")
        model = AutoModel.from_pretrained(MODEL_ID)
        model.save_pretrained(MODEL_DIR)
        
        # 모델 크기 계산
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✅ 총 파라미터 수: {total_params:,} ({total_params/1e6:.1f}M)")
        
        # 4. RBE 설정 파일 생성
        print("\n⚙️  RBE 설정 파일 생성 중...")
        rbe_config = {
            "model_info": {
                "name": "KoMiniLM-23M",
                "model_id": MODEL_ID,
                "language": "korean",
                "total_parameters": total_params,
                "model_type": config.model_type,
                "architecture": "bert"
            },
            "rbe_settings": {
                "compression_target": {
                    "quality_grade": "A",  # 고품질로 시작
                    "target_compression_ratio": 500.0,  # 500:1 압축 목표
                    "max_rmse": 1e-4
                },
                "poincare_ball": {
                    "dimension": 128,  # 128비트 압축
                    "coordinate_precision": 112,  # 각도 좌표 정밀도
                    "radius_precision": 15  # 반지름 정밀도
                },
                "bit_differential": {
                    "cycle_length": 2048,  # 11비트 사이클
                    "num_bit_planes": 128,
                    "error_threshold": 1e-6,
                    "stability_check": True
                }
            },
            "test_settings": {
                "max_sequence_length": 128,
                "batch_size": 4,
                "test_prompts": [
                    "안녕하세요",
                    "오늘 날씨가 좋네요",
                    "RBE 압축 시스템을 테스트합니다",
                    "한국어 자연어처리 모델입니다"
                ]
            }
        }
        
        with open(MODEL_DIR / "rbe_config.json", "w", encoding="utf-8") as f:
            json.dump(rbe_config, f, ensure_ascii=False, indent=2)
        
        # 5. 가중치 추출 및 분석
        print("\n🔍 가중치 분석 중...")
        weight_analysis = analyze_model_weights(model)
        
        with open(MODEL_DIR / "weight_analysis.json", "w", encoding="utf-8") as f:
            json.dump(weight_analysis, f, indent=2)
        
        print(f"  ✅ 레이어 수: {weight_analysis['num_layers']}")
        print(f"  ✅ 가중치 분포:")
        for layer_type, stats in weight_analysis['layer_stats'].items():
            print(f"    {layer_type}: 평균={stats['mean']:.6f}, 표준편차={stats['std']:.6f}")
        
        # 6. 간단한 추론 테스트
        print("\n🧪 추론 테스트 중...")
        model.eval()
        with torch.no_grad():
            test_inputs = tokenizer(
                rbe_config["test_settings"]["test_prompts"], 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            )
            
            outputs = model(**test_inputs)
            print(f"  ✅ 출력 형태: {outputs.last_hidden_state.shape}")
            print(f"  ✅ 추론 성공: {len(rbe_config['test_settings']['test_prompts'])}개 프롬프트")
        
        print("\n🎉 한국어 테스트 모델 설정 완료!")
        print(f"📍 모델 경로: {MODEL_DIR}")
        print(f"📊 압축 목표: {rbe_config['rbe_settings']['compression_target']['target_compression_ratio']}:1")
        print(f"🎯 다음 단계: RBETensor 구현 및 모델 압축")
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

def analyze_model_weights(model):
    """모델 가중치 분석"""
    
    layer_stats = {}
    total_params = 0
    
    for name, param in model.named_parameters():
        layer_type = name.split('.')[0] if '.' in name else name
        
        if layer_type not in layer_stats:
            layer_stats[layer_type] = {
                'count': 0,
                'total_params': 0,
                'values': []
            }
        
        layer_stats[layer_type]['count'] += 1
        layer_stats[layer_type]['total_params'] += param.numel()
        layer_stats[layer_type]['values'].extend(param.flatten().tolist()[:1000])  # 샘플링
        total_params += param.numel()
    
    # 통계 계산
    for layer_type in layer_stats:
        values = layer_stats[layer_type]['values']
        if values:
            import statistics
            layer_stats[layer_type]['mean'] = statistics.mean(values)
            layer_stats[layer_type]['std'] = statistics.stdev(values) if len(values) > 1 else 0.0
            layer_stats[layer_type]['min'] = min(values)
            layer_stats[layer_type]['max'] = max(values)
            del layer_stats[layer_type]['values']  # 메모리 절약
    
    return {
        'total_parameters': total_params,
        'num_layers': len(layer_stats),
        'layer_stats': layer_stats
    }

if __name__ == "__main__":
    try:
        success = setup_korean_test_model()
        if success:
            print("\n✨ 준비 완료! 이제 RBE 구현을 시작할 수 있습니다.")
        else:
            print("\n💥 설정 실패. 다시 시도해주세요.")
    except Exception as e:
        print(f"❌ 전체 오류: {e}")
        print("필요한 라이브러리 설치: pip install transformers torch") 