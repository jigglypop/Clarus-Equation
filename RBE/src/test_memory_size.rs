use rbe_llm::packed_params::{PoincarePackedBit128, HybridEncodedBlock, Packed128, ResidualCoefficient};

fn main() {
    println!("=== RBE 라이브러리 메모리 크기 분석 ===\n");
    
    // 기본 128비트 구조체
    println!("[ 기본 128비트 구조체 ]");
    println!("Packed128: {} bytes", std::mem::size_of::<Packed128>());
    println!("PoincarePackedBit128: {} bytes", std::mem::size_of::<PoincarePackedBit128>());
    
    // 잔차 계수
    println!("\n[ 잔차 계수 ]");
    println!("ResidualCoefficient: {} bytes", std::mem::size_of::<ResidualCoefficient>());
    
    // 전체 블록 구조체
    println!("\n[ 인코딩된 블록 ]");
    println!("HybridEncodedBlock (기본): {} bytes", std::mem::size_of::<HybridEncodedBlock>());
    
    // 실제 블록 크기 계산 (예시)
    let test_block = HybridEncodedBlock {
        rbe_params: [0.0; 8],
        residuals: vec![],
        rows: 64,
        cols: 64,
        transform_type: rbe_llm::TransformType::Dwt,
    };
    
    let base_size = std::mem::size_of::<HybridEncodedBlock>();
    let residuals_size = test_block.residuals.capacity() * std::mem::size_of::<ResidualCoefficient>();
    
    println!("\n[ 실제 메모리 사용량 (64x64 블록 예시) ]");
    println!("기본 구조체 크기: {} bytes", base_size);
    println!("잔차 계수 벡터 용량: {} bytes", residuals_size);
    println!("총 크기: {} bytes", base_size + residuals_size);
    
    // 압축률 계산
    let original_size = 64 * 64 * 4; // f32 = 4 bytes
    let compression_ratio = original_size as f32 / base_size as f32;
    
    println!("\n[ 압축률 분석 ]");
    println!("원본 64x64 f32 행렬: {} bytes", original_size);
    println!("압축된 블록 (잔차 없이): {} bytes", base_size);
    println!("압축률: {:.1}x", compression_ratio);
    
    // 다양한 잔차 개수에 따른 실제 크기
    println!("\n[ 잔차 개수에 따른 실제 크기 ]");
    for k in [0, 8, 40, 148, 500] {
        let total_size = base_size + k * std::mem::size_of::<ResidualCoefficient>();
        let ratio = original_size as f32 / total_size as f32;
        println!("K={:3}: {:5} bytes, 압축률: {:.1}x", k, total_size, ratio);
    }
} 