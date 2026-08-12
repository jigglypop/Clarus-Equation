use rbe_llm::core::{
    encoder::RBEEncoder,
    decoder::weight_generator::WeightGenerator,
    packed_params::TransformType,
};
use std::time::Instant;

fn main() {
    println!("🚀 RBE 압축 기술 데모\n");
    
    // 1. 작은 행렬로 압축 효과 시연
    println!("📊 1. 작은 행렬 압축 데모 (16x16)");
    println!("{}", "=".repeat(50));
    
    // 테스트 데이터 생성 (사인파 패턴)
    let size = 16;
    let mut data = vec![0.0f32; size * size];
    for i in 0..size {
        for j in 0..size {
            let x = i as f32 / size as f32 * std::f32::consts::PI * 2.0;
            let y = j as f32 / size as f32 * std::f32::consts::PI * 2.0;
            data[i * size + j] = (x.sin() + y.cos()) * 0.5;
        }
    }
    
    // 원본 데이터 일부 출력
    println!("원본 데이터 (첫 4x4):");
    for i in 0..4 {
        for j in 0..4 {
            print!("{:6.3} ", data[i * size + j]);
        }
        println!();
    }
    
    // RBE 인코딩
    let mut encoder = RBEEncoder::new(8, TransformType::Dct);
    let start = Instant::now();
    let encoded = encoder.encode_block(&data, size, size);
    let encode_time = start.elapsed();
    
    println!("\n✅ 인코딩 완료!");
    println!("  - 원본 크기: {} bytes", size * size * 4);
    println!("  - 압축 크기: {} bytes (RBE 파라미터 8개)", 
             32 + 8 + 8 + 1 + 24); // 실제 구조체 크기
    println!("  - 압축률: {:.1}x", (size * size * 4) as f32 / 73.0);
    println!("  - 인코딩 시간: {:?}", encode_time);
    
    // RBE 파라미터 출력
    println!("\n🔍 RBE 파라미터 (8개):");
    println!("  Params: {:?}", &encoded.rbe_params[..4]);
    println!("  More params: {:?}", &encoded.rbe_params[4..]);
    
    // 디코딩
    let weight_generator = WeightGenerator::new();
    let start = Instant::now();
    let decoded = weight_generator.decode_block(&encoded);
    let decode_time = start.elapsed();
    
    println!("\n✅ 디코딩 완료!");
    println!("  - 디코딩 시간: {:?}", decode_time);
    
    // 복원된 데이터 일부 출력
    println!("\n복원된 데이터 (첫 4x4):");
    for i in 0..4 {
        for j in 0..4 {
            print!("{:6.3} ", decoded[i * size + j]);
        }
        println!();
    }
    
    // RMSE 계산
    let mut sum_sq_diff = 0.0;
    for i in 0..data.len() {
        let diff = data[i] - decoded[i];
        sum_sq_diff += diff * diff;
    }
    let rmse = (sum_sq_diff / data.len() as f32).sqrt();
    println!("\n📏 복원 정확도 (RMSE): {:.6}", rmse);
    
    // 2. 큰 행렬 압축 시연
    println!("\n\n📊 2. 큰 행렬 압축 (256x256)");
    println!("{}", "=".repeat(50));
    
    let big_size = 256;
    let mut big_data = vec![0.0f32; big_size * big_size];
    
    // 그라디언트 패턴 생성
    for i in 0..big_size {
        for j in 0..big_size {
            big_data[i * big_size + j] = (i as f32 / big_size as f32) * (j as f32 / big_size as f32);
        }
    }
    
    // 여러 K값으로 압축률 비교
    println!("\nK값에 따른 압축률 비교:");
    for k in [8, 16, 32, 64] {
        let mut encoder = RBEEncoder::new(k, TransformType::Dct);
        let start = Instant::now();
        let encoded = encoder.encode_block(&big_data, big_size, big_size);
        let encode_time = start.elapsed();
        
        // 디코딩 후 RMSE 계산
        let decoded = weight_generator.decode_block(&encoded);
        let mut sum_sq = 0.0;
        for i in 0..big_data.len() {
            let diff = big_data[i] - decoded[i];
            sum_sq += diff * diff;
        }
        let rmse = (sum_sq / big_data.len() as f32).sqrt();
        
        // 압축 크기 계산
        let compressed_size = 73 + encoded.residuals.len() * 8;
        let compression_ratio = (big_size * big_size * 4) as f32 / compressed_size as f32;
        
        println!("\n  K={:3}: 압축률 {:6.1}x, RMSE {:.6}, 시간 {:?}", 
                 k, compression_ratio, rmse, encode_time);
    }
    
    // 3. 성능 벤치마크
    println!("\n\n⚡ 성능 벤치마크 (64x64 행렬, 100회 반복)");
    println!("{}", "=".repeat(50));
    
    let bench_size = 64;
    let bench_data = vec![0.1f32; bench_size * bench_size];
    let mut encoder = RBEEncoder::new(32, TransformType::Dct);
    
    // 인코딩 성능
    let start = Instant::now();
    for _ in 0..100 {
        let _ = encoder.encode_block(&bench_data, bench_size, bench_size);
    }
    let total_encode = start.elapsed();
    
    // 디코딩 성능
    let encoded = encoder.encode_block(&bench_data, bench_size, bench_size);
    let start = Instant::now();
    for _ in 0..100 {
        let _ = weight_generator.decode_block(&encoded);
    }
    let total_decode = start.elapsed();
    
    println!("  - 평균 인코딩 시간: {:?}", total_encode / 100);
    println!("  - 평균 디코딩 시간: {:?}", total_decode / 100);
    println!("  - 디코딩 처리량: {:.2} MB/s", 
             (bench_size * bench_size * 4 * 100) as f64 / 1024.0 / 1024.0 / total_decode.as_secs_f64());
    
    println!("\n✨ RBE 압축 기술로 빠른 인코딩/디코딩과 높은 압축률을 동시에 달성합니다!");
} 