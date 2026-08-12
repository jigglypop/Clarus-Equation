# RBETensor 설계 계획서

## 🎯 핵심 개념

**RBETensor는 블록의 겹침 표현이다**
- 텐서의 각 영역을 `HybridEncodedBlock`으로 분할
- 논리적 인덱싱 = 블록 찾기 + 블록 내 위치 계산
- 연산 = 블록 단위 연산 + 결과 블록들의 조합

## 📐 설계 원칙

### 1. 블록 중심 구조
```rust
pub struct RBETensor {
    // 논리적 형태
    pub logical_shape: Vec<usize>,
    
    // 블록 구조 (핵심!)
    pub blocks: Vec<HybridEncodedBlock>,    // 기존 core 구현체 활용
    pub block_layout: BlockLayout,          // 블록들의 배치 정보
    
    // 메타데이터
    pub requires_grad: bool,
    pub compression_metadata: CompressionMetadata,
}
```

### 2. 블록 레이아웃 관리
```rust
pub struct BlockLayout {
    pub block_shape: Vec<usize>,      // 각 블록의 크기 [32, 32]
    pub grid_shape: Vec<usize>,       // 블록 격자 크기 [2, 2]
    pub total_blocks: usize,          // 총 블록 개수
    pub block_strides: Vec<usize>,    // 블록 간 stride
}
```

### 3. 인덱싱 전략
```rust
impl RBETensor {
    // 논리적 인덱스 → (블록 ID, 블록 내 인덱스)
    fn logical_to_block_coordinate(&self, indices: &[usize]) -> (usize, Vec<usize>);
    
    // 블록 ID → 해당 블록의 논리적 시작 위치
    fn block_to_logical_start(&self, block_id: usize) -> Vec<usize>;
}
```

## 🔧 구현 단계

### Phase 1: 기본 블록 텐서 (1주)
1. **BlockLayout 구조체 구현**
   - 블록 분할 알고리즘
   - 인덱스 변환 함수들

2. **RBETensor 기본 구조**
   - 기존 `HybridEncodedBlock` 활용
   - 생성자: `from_data()`, `zeros()`, `ones()`

3. **기본 접근 연산**
   - `get()`: 블록 찾기 + 블록 내 값 추출
   - `set()`: 블록 찾기 + 블록 내 값 수정

### Phase 2: 블록 단위 연산 (1주)
1. **블록별 독립 연산**
   - `add()`: 대응 블록끼리 덧셈
   - `mul()`: 대응 블록끼리 곱셈

2. **기존 core 연산 활용**
   - `WeightGenerator`로 블록 복원
   - `RBEEncoder`로 결과 재압축

### Phase 3: 고급 연산 (2주)
1. **행렬 곱셈 (`matmul`)**
   - 블록 단위 GEMM
   - 결과 블록 조합

2. **텐서 변형**
   - `reshape()`: 블록 재배치
   - `transpose()`: 블록 순서 변경

## 📊 예시: 2x2 텐서의 블록 표현

### 입력 데이터
```rust
let data = vec![1.0, 2.0, 3.0, 4.0];
let shape = vec![2, 2];
```

### 블록 분할 (block_size = 1)
```
원본 텐서:     블록 구조:
[1, 2]   →    [Block0] [Block1]
[3, 4]        [Block2] [Block3]

Block0: HybridEncodedBlock { data: [1.0], pos: (0,0) }
Block1: HybridEncodedBlock { data: [2.0], pos: (0,1) }  
Block2: HybridEncodedBlock { data: [3.0], pos: (1,0) }
Block3: HybridEncodedBlock { data: [4.0], pos: (1,1) }
```

### 인덱싱 예시
```rust
tensor.get(&[1, 0]) → 
1. 논리적 [1,0] → 블록 ID = 2, 블록 내 [0,0]
2. blocks[2].decode_at([0,0]) → 3.0
```

## 🎛️ 기존 Core 연결점

### 1. 블록 생성/압축
```rust
// 기존 RBEEncoder 활용
let mut encoder = RBEEncoder::new(config);
let block = encoder.encode_block(&block_data, block_rows, block_cols)?;
```

### 2. 블록 복원/추론
```rust
// 기존 WeightGenerator 활용  
let mut generator = WeightGenerator::new();
let values = generator.decode_block(&block)?;
```

### 3. 직접 추론 (압축 해제 없이)
```rust
// 기존 GridDirectInference 활용
let inference = GridDirectInference::new(grid_rows, grid_cols, block_size);
let value = inference.infer_at_position(row, col)?;
```

## 🧪 테스트 전략

### 1. 블록 분할 정확성
- 다양한 텐서 크기 × 블록 크기 조합
- 인덱스 변환 양방향 검증

### 2. 블록 연산 정확성  
- 블록별 연산 결과 = 전체 텐서 연산 결과
- 압축/복원 과정의 오차 측정

### 3. 한국어 모델 호환성
- KoMiniLM-23M 가중치 블록 분할
- 실제 추론 정확도 검증

## 📈 성능 목표

### 압축 효율성
- 목표: 500:1 압축 비율
- 품질: A등급 (RMSE < 1e-4)

### 연산 성능
- 블록 단위 병렬 처리
- 메모리 효율적 스트리밍

### 한국어 모델 실용성
- 23M 파라미터 → 46KB 압축
- 실시간 추론 가능

## 🚀 다음 단계

1. **BlockLayout 구현** (우선순위 1)
2. **기본 RBETensor 구조** (우선순위 1) 
3. **블록 인덱싱 시스템** (우선순위 1)
4. **기존 core 연결** (우선순위 2)
5. **한국어 모델 테스트** (우선순위 3)

---

**핵심**: RBETensor = HybridEncodedBlock들의 스마트한 조합
**목표**: 기존 core 100% 활용하여 텐서 추상화 제공 