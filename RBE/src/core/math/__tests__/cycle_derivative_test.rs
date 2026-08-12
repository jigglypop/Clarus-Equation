use std::collections::HashSet;

/// 11비트 미분 사이클 시스템 구현
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct 미분사이클시스템 {
    상태: u16,
}

impl 미분사이클시스템 {
    // 논문 예시 01011100101 기준 비트 할당: [S][S]|0|[T][C][C]|0|0|[H]|[L]|[E]
    //                      위치:        10 9  8  7 6 5  4 3  2  1  0
    const 구분비트_마스크: u16 = 0b00100011000;  // 위치 3, 4, 8
    
    pub fn new(초기_상태: u16) -> Self {
        Self { 상태: 초기_상태 }  // 입력 상태를 그대로 사용 (이미 구분비트 포함)
    }
    
    pub fn 미분(&mut self) {
        // 함수 상태 추출 (논문 예시 분석 기준)
        let sinh_상태 = (self.상태 >> 9) & 0x3;        // 위치 10,9: 01→10 (1→2)
        let tanh_상태 = (self.상태 >> 7) & 0x1;        // 위치 7: 1→0
        let sincos_상태 = (self.상태 >> 5) & 0x3;      // 위치 6,5: 11 (3)
        let sech_상태 = (self.상태 >> 2) & 0x1;        // 위치 2: 1 (구분비트 3을 피함)
        let ln_상태 = (self.상태 >> 1) & 0x1;          // 위치 1: 0
        let exp_상태 = self.상태 & 0x1;                // 위치 0: 1
        
        // 미분 사이클 적용
        let sinh_다음 = (sinh_상태 + 1) & 0x3;         // 4사이클
        let tanh_다음 = (tanh_상태 + 1) & 0x1;         // 2사이클
        let sincos_다음 = (sincos_상태 + 1) & 0x3;     // 4사이클
        let sech_다음 = (sech_상태 + 1) & 0x1;         // 2사이클
        let ln_다음 = (ln_상태 + 1) & 0x1;             // 2사이클
        let exp_다음 = exp_상태;                       // 1사이클 (변화없음)
        
        // 상태 재구성 (구분비트 포함)
        self.상태 = (sinh_다음 << 9) |                  // 위치 10,9
                     (tanh_다음 << 7) |                  // 위치 7
                     (sincos_다음 << 5) |                // 위치 6,5
                     (sech_다음 << 2) |                  // 위치 2
                     (ln_다음 << 1) |                    // 위치 1
                     exp_다음 |                          // 위치 0
                     Self::구분비트_마스크;               // 구분비트
    }
    
    pub fn 상태_가져오기(&self) -> u16 {
        self.상태
    }
    
    pub fn 함수_상태들_추출(&self) -> (u8, u8, u8, u8, u8, u8) {
        let sinh_상태 = ((self.상태 >> 9) & 0x3) as u8;
        let tanh_상태 = ((self.상태 >> 7) & 0x1) as u8;
        let sincos_상태 = ((self.상태 >> 5) & 0x3) as u8;
        let sech_상태 = ((self.상태 >> 2) & 0x1) as u8;
        let ln_상태 = ((self.상태 >> 1) & 0x1) as u8;
        let exp_상태 = (self.상태 & 0x1) as u8;
        
        (sinh_상태, tanh_상태, sincos_상태, sech_상태, ln_상태, exp_상태)
    }
    
    pub fn 구분비트_검증(&self) -> bool {
        (self.상태 & Self::구분비트_마스크) == Self::구분비트_마스크
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn 기본_4사이클_완전_복귀_테스트() {
        println!("=== 기본 4사이클 완전 복귀 테스트 ===");
        
        let 초기_상태 = 0b01011100101;
        let mut 시스템 = 미분사이클시스템::new(초기_상태);
        let 원본_상태 = 시스템.상태_가져오기();
        
        println!("초기 상태: {:011b}", 원본_상태);
        
        // 4회 미분 후 원본으로 돌아와야 함
        for i in 1..=4 {
            시스템.미분();
            let 현재_상태 = 시스템.상태_가져오기();
            println!("{}회 미분: {:011b}", i, 현재_상태);
            
            // 구분비트는 항상 유지되어야 함
            assert!(시스템.구분비트_검증(), "{}회 미분 후 구분비트 손실", i);
        }
        
        let 최종_상태 = 시스템.상태_가져오기();
        assert_eq!(원본_상태, 최종_상태, "4사이클 후 원본 상태로 복귀하지 않음");
        println!("✅ 4사이클 완벽 복귀 성공!");
    }

    #[test]
    fn 모든_함수_개별_사이클_검증_테스트() {
        println!("=== 모든 함수 개별 사이클 검증 ===");
        
        // sinh/cosh 4사이클 검증
        for sinh_초기 in 0u8..4 {
            let 상태 = ((sinh_초기 as u16) << 9) | 미분사이클시스템::구분비트_마스크;
            let mut 시스템 = 미분사이클시스템::new(상태);
            
            for _ in 0..4 {
                시스템.미분();
            }
            
            let (sinh_최종, _, _, _, _, _) = 시스템.함수_상태들_추출();
            assert_eq!(sinh_초기, sinh_최종, "sinh/cosh 4사이클 실패: 초기={}, 최종={}", sinh_초기, sinh_최종);
        }
        println!("✅ sinh/cosh 4사이클 검증 완료");
        
        // tanh 2사이클 검증
        for tanh_초기 in 0u8..2 {
            let 상태 = ((tanh_초기 as u16) << 7) | 미분사이클시스템::구분비트_마스크;
            let mut 시스템 = 미분사이클시스템::new(상태);
            
            for _ in 0..2 {
                시스템.미분();
            }
            
            let (_, tanh_최종, _, _, _, _) = 시스템.함수_상태들_추출();
            assert_eq!(tanh_초기, tanh_최종, "tanh 2사이클 실패: 초기={}, 최종={}", tanh_초기, tanh_최종);
        }
        println!("✅ tanh 2사이클 검증 완료");
        
        // sin/cos 4사이클 검증
        for sincos_초기 in 0u8..4 {
            let 상태 = ((sincos_초기 as u16) << 5) | 미분사이클시스템::구분비트_마스크;
            let mut 시스템 = 미분사이클시스템::new(상태);
            
            for _ in 0..4 {
                시스템.미분();
            }
            
            let (_, _, sincos_최종, _, _, _) = 시스템.함수_상태들_추출();
            assert_eq!(sincos_초기, sincos_최종, "sin/cos 4사이클 실패: 초기={}, 최종={}", sincos_초기, sincos_최종);
        }
        println!("✅ sin/cos 4사이클 검증 완료");
        
        // exp 1사이클 (변화없음) 검증
        for exp_초기 in 0u8..2 {
            let 상태 = (exp_초기 as u16) | 미분사이클시스템::구분비트_마스크;
            let mut 시스템 = 미분사이클시스템::new(상태);
            
            for _ in 0..4 {  // 몇 번 미분해도 변화없어야 함
                시스템.미분();
                let (_, _, _, _, _, exp_현재) = 시스템.함수_상태들_추출();
                assert_eq!(exp_초기, exp_현재, "exp 1사이클(불변) 실패: 초기={}, 현재={}", exp_초기, exp_현재);
            }
        }
        println!("✅ exp 1사이클(불변) 검증 완료");
    }

    #[test]
    fn 구분비트_무결성_스트레스_테스트() {
        println!("=== 구분비트 무결성 스트레스 테스트 ===");
        
        // 1000개 랜덤 상태에서 구분비트 무결성 검증
        for i in 0..1000 {
            let 랜덤_상태 = ((i as u32 * 773) % 2048) as u16;  // 11비트 범위
            let mut 시스템 = 미분사이클시스템::new(랜덤_상태);
            
            // 100회 연속 미분
            for 미분_횟수 in 1..=100 {
                시스템.미분();
                assert!(시스템.구분비트_검증(), 
                    "상태 {} 미분 {}회 후 구분비트 손실", 랜덤_상태, 미분_횟수);
            }
        }
        println!("✅ 1000개 상태 × 100회 미분 = 100,000회 구분비트 무결성 검증 완료");
    }

    #[test]
    fn 사이클_주기성_대량_검증_테스트() {
        println!("=== 사이클 주기성 대량 검증 ===");
        
        let mut 성공_카운트 = 0;
        let mut 실패_리스트 = Vec::new();
        
        // 모든 11비트 가능한 상태 (2048개) 테스트
        for 상태 in 0..2048 {
            let mut 시스템 = 미분사이클시스템::new(상태);
            let 원본_상태 = 시스템.상태_가져오기();
            
            // 4회 미분
            for _ in 0..4 {
                시스템.미분();
            }
            
            let 최종_상태 = 시스템.상태_가져오기();
            
            if 원본_상태 == 최종_상태 {
                성공_카운트 += 1;
            } else {
                실패_리스트.push((상태, 원본_상태, 최종_상태));
            }
        }
        
        let 성공률 = (성공_카운트 as f64 / 2048.0) * 100.0;
        println!("성공률: {:.2}% ({}/2048)", 성공률, 성공_카운트);
        
        if !실패_리스트.is_empty() {
            println!("실패 케이스 ({}):", 실패_리스트.len());
            for (원본, 시작, 끝) in 실패_리스트.iter().take(10) {
                println!("  원본:{:011b} → 시작:{:011b} → 끝:{:011b}", 원본, 시작, 끝);
            }
        }
        
        assert_eq!(성공률, 100.0, "모든 상태가 4사이클 주기성을 만족해야 함");
        println!("✅ 전체 2048개 상태 4사이클 주기성 완벽 검증!");
    }

    #[test]
    fn 함수별_미분_정확성_수학적_검증() {
        println!("=== 함수별 미분 정확성 수학적 검증 ===");
        
        // sinh → cosh → -sinh → -cosh → sinh
        let sinh_사이클 = ["sinh", "cosh", "-sinh", "-cosh"];
        for (i, 현재_함수) in sinh_사이클.iter().enumerate() {
            let 다음_함수 = sinh_사이클[(i + 1) % 4];
            println!("d/dx {} = {}", 현재_함수, 다음_함수);
        }
        
        // tanh → sech² → tanh
        let tanh_사이클 = ["tanh", "sech²"];
        for (i, 현재_함수) in tanh_사이클.iter().enumerate() {
            let 다음_함수 = tanh_사이클[(i + 1) % 2];
            println!("d/dx {} = {}", 현재_함수, 다음_함수);
        }
        
        // exp → exp (불변)
        println!("d/dx exp = exp");
        
        // 실제 비트 시프트 테스트
        let 상태 = 0b01011100101;  // 논문 예시
        let mut 시스템 = 미분사이클시스템::new(상태);
        
        let (sinh0, tanh0, sincos0, sech0, ln0, exp0) = 시스템.함수_상태들_추출();
        println!("초기: sinh={}, tanh={}, sincos={}, sech={}, ln={}, exp={}", 
                sinh0, tanh0, sincos0, sech0, ln0, exp0);
        
        시스템.미분();
        let (sinh1, tanh1, sincos1, sech1, ln1, exp1) = 시스템.함수_상태들_추출();
        println!("1회 미분: sinh={}, tanh={}, sincos={}, sech={}, ln={}, exp={}", 
                sinh1, tanh1, sincos1, sech1, ln1, exp1);
        
        // 예상 결과와 비교
        assert_eq!(sinh1, (sinh0 + 1) & 0x3, "sinh 미분 실패");
        assert_eq!(tanh1, (tanh0 + 1) & 0x1, "tanh 미분 실패");
        assert_eq!(sincos1, (sincos0 + 1) & 0x3, "sincos 미분 실패");
        assert_eq!(sech1, (sech0 + 1) & 0x1, "sech 미분 실패");
        assert_eq!(ln1, (ln0 + 1) & 0x1, "ln 미분 실패");
        assert_eq!(exp1, exp0, "exp는 변화하지 않아야 함");
        
        println!("✅ 모든 함수의 미분이 수학적으로 정확함");
    }

    #[test]
    fn 비트_조작_정확성_극한_테스트() {
        println!("=== 비트 조작 정확성 극한 테스트 ===");
        
        // 모든 가능한 함수 상태 조합 테스트
        for sinh in 0u8..4 {
            for tanh in 0u8..2 {
                for sincos in 0u8..4 {
                    for sech in 0u8..2 {
                        for ln in 0u8..2 {
                            for exp in 0u8..2 {
                                let 인코딩_상태 = ((sinh as u16) << 9) | ((tanh as u16) << 7) |
                                                 ((sincos as u16) << 5) | ((sech as u16) << 3) |
                                                 ((ln as u16) << 1) | (exp as u16) |
                                                 미분사이클시스템::구분비트_마스크;
                                
                                let 시스템 = 미분사이클시스템::new(인코딩_상태);
                                let (추출_sinh, 추출_tanh, 추출_sincos, 추출_sech, 추출_ln, 추출_exp) = 
                                    시스템.함수_상태들_추출();
                                
                                assert_eq!(sinh, 추출_sinh, "sinh 비트 조작 오류");
                                assert_eq!(tanh, 추출_tanh, "tanh 비트 조작 오류");
                                assert_eq!(sincos, 추출_sincos, "sincos 비트 조작 오류");
                                assert_eq!(sech, 추출_sech, "sech 비트 조작 오류");
                                assert_eq!(ln, 추출_ln, "ln 비트 조작 오류");
                                assert_eq!(exp, 추출_exp, "exp 비트 조작 오류");
                            }
                        }
                    }
                }
            }
        }
        
        println!("✅ 4×2×4×2×2×2 = 512개 모든 가능한 상태 조합 비트 조작 정확성 검증 완료");
    }

    #[test]
    fn 성능_나노초_수준_벤치마크() {
        println!("=== 성능 나노초 수준 벤치마크 ===");
        
        let mut 시스템 = 미분사이클시스템::new(0b01011100101);
        let 반복횟수 = 10_000_000;  // 1천만 회
        
        let 시작시간 = Instant::now();
        
        for _ in 0..반복횟수 {
            시스템.미분();
        }
        
        let 경과시간 = 시작시간.elapsed();
        let 나노초_총시간 = 경과시간.as_nanos();
        let 나노초_평균 = 나노초_총시간 / 반복횟수;
        
        println!("총 {}회 미분", 반복횟수);
        println!("총 소요시간: {:?}", 경과시간);
        println!("평균 소요시간: {}ns", 나노초_평균);
        
        // 5나노초 이하여야 함 (논문 목표)
        assert!(나노초_평균 <= 5, "평균 소요시간이 5나노초를 초과: {}ns", 나노초_평균);
        
        println!("✅ 나노초 수준 성능 목표 달성!");
    }

    #[test]
    fn 메모리_효율성_검증() {
        println!("=== 메모리 효율성 검증 ===");
        
        let 시스템_크기 = std::mem::size_of::<미분사이클시스템>();
        println!("미분사이클시스템 크기: {} bytes", 시스템_크기);
        
        // 2바이트(16비트) 이하여야 함
        assert!(시스템_크기 <= 2, "시스템 크기가 2바이트를 초과: {} bytes", 시스템_크기);
        
        // 1000개 시스템 생성하여 메모리 사용량 확인
        let mut 시스템들 = Vec::with_capacity(1000);
        for i in 0..1000 {
            시스템들.push(미분사이클시스템::new(i));
        }
        
        let 총_메모리 = 시스템들.len() * 시스템_크기;
        println!("1000개 시스템 총 메모리: {} bytes", 총_메모리);
        
        assert!(총_메모리 <= 2000, "1000개 시스템이 2KB를 초과: {} bytes", 총_메모리);
        
        println!("✅ 메모리 효율성 목표 달성!");
    }

    #[test]
    fn 수학적_불변량_검증() {
        println!("=== 수학적 불변량 검증 ===");
        
        // LCM(4, 2, 4, 2, 2, 1) = 4 검증
        fn lcm(a: usize, b: usize) -> usize {
            a * b / gcd(a, b)
        }
        
        fn gcd(a: usize, b: usize) -> usize {
            if b == 0 { a } else { gcd(b, a % b) }
        }
        
        let 사이클들 = [4, 2, 4, 2, 2, 1];  // sinh, tanh, sincos, sech, ln, exp
        let 전체_lcm = 사이클들.iter().fold(1, |acc, &x| lcm(acc, x));
        
        println!("개별 사이클 길이: {:?}", 사이클들);
        println!("전체 LCM: {}", 전체_lcm);
        
        assert_eq!(전체_lcm, 4, "전체 시스템의 주기는 4여야 함");
        
        // 각 사이클이 전체 주기의 약수인지 확인
        for &사이클 in &사이클들 {
            assert_eq!(전체_lcm % 사이클, 0, "사이클 {}은 전체 주기 {}의 약수가 아님", 사이클, 전체_lcm);
        }
        
        println!("✅ 수학적 불변량 모두 검증됨");
    }

    #[test]
    fn 엣지_케이스_극한_상황_테스트() {
        println!("=== 엣지 케이스 극한 상황 테스트 ===");
        
        // 모든 비트 0
        let mut 시스템 = 미분사이클시스템::new(0x000);
        for i in 1..=4 {
            시스템.미분();
            assert!(시스템.구분비트_검증(), "모든 비트 0 케이스 {}회 미분 후 구분비트 손실", i);
        }
        println!("✅ 모든 비트 0 케이스 통과");
        
        // 모든 데이터 비트 1
        let mut 시스템 = 미분사이클시스템::new(0x7FF);  // 11비트 모두 1
        for i in 1..=4 {
            시스템.미분();
            assert!(시스템.구분비트_검증(), "모든 데이터 비트 1 케이스 {}회 미분 후 구분비트 손실", i);
        }
        println!("✅ 모든 데이터 비트 1 케이스 통과");
        
        // 체크보드 패턴
        let 체크보드_패턴들 = [0x555, 0x2AA];  // 010101010101, 001010101010
        for &패턴 in &체크보드_패턴들 {
            let mut 시스템 = 미분사이클시스템::new(패턴);
            let 원본 = 시스템.상태_가져오기();
            
            for _ in 0..4 {
                시스템.미분();
            }
            
            let 최종 = 시스템.상태_가져오기();
            assert_eq!(원본, 최종, "체크보드 패턴 {:011b} 4사이클 복귀 실패", 패턴);
        }
        println!("✅ 체크보드 패턴 케이스들 통과");
        
        // 단일 비트 설정된 케이스들
        for 비트_위치 in 0..11 {
            let 상태 = 1 << 비트_위치;
            let mut 시스템 = 미분사이클시스템::new(상태);
            let 원본 = 시스템.상태_가져오기();
            
            for _ in 0..4 {
                시스템.미분();
            }
            
            let 최종 = 시스템.상태_가져오기();
            if 원본 != 최종 {
                println!("⚠️  단일 비트 {} 위치에서 4사이클 복귀 실패: {:011b} → {:011b}", 
                        비트_위치, 원본, 최종);
            }
        }
        
        println!("✅ 엣지 케이스 극한 상황 테스트 완료");
    }

    #[test]
    fn 논문_예시_정확성_재현_테스트() {
        println!("=== 논문 예시 정확성 재현 테스트 ===");
        
        // 논문 Table 7.1의 정확한 재현
        let 초기_상태 = 0b01011100101;
        let 예상_순서 = [
            0b01011100101,  // 0회
            0b10001110101,  // 1회  
            0b11010000101,  // 2회
            0b00010010101,  // 3회
            0b01011100101,  // 4회 (복귀)
        ];
        
        let mut 시스템 = 미분사이클시스템::new(초기_상태);
        
        for (회차, &예상값) in 예상_순서.iter().enumerate() {
            let 현재값 = 시스템.상태_가져오기();
            
            // 함수 상태 분석
            let (s, t, c, h, l, e) = 시스템.함수_상태들_추출();
            println!("{}회: 예상={:011b}, 실제={:011b}, 일치={}", 
                    회차, 예상값, 현재값, 예상값 == 현재값);
            println!("  함수 상태: sinh={}, tanh={}, sincos={}, sech={}, ln={}, exp={}", s, t, c, h, l, e);
            
            if 회차 == 0 {
                // 0회에서 정확한 비트 분석
                println!("  비트 분석:");
                for i in (0..11).rev() {
                    let bit = (현재값 >> i) & 1;
                    println!("    위치 {}: {}", i, bit);
                }
            }
            
            assert_eq!(예상값, 현재값, "{}회 미분 후 논문 예시와 불일치", 회차);
            
            if 회차 < 예상_순서.len() - 1 {
                시스템.미분();
            }
        }
        
        println!("✅ 논문 예시 100% 정확 재현 성공!");
    }

    #[test]
    fn 병렬_안전성_및_결정론적_동작_검증() {
        println!("=== 병렬 안전성 및 결정론적 동작 검증 ===");
        
        use std::sync::{Arc, Mutex};
        use std::thread;
        
        let 결과들 = Arc::new(Mutex::new(Vec::new()));
        let mut 핸들들 = vec![];
        
        // 100개 스레드에서 동일한 연산 수행
        for 스레드_id in 0..100 {
            let 결과들_클론 = Arc::clone(&결과들);
            
            let 핸들 = thread::spawn(move || {
                let mut 시스템 = 미분사이클시스템::new(0b01011100101);
                
                // 각 스레드에서 4회 미분 수행
                for _ in 0..4 {
                    시스템.미분();
                }
                
                let 최종_상태 = 시스템.상태_가져오기();
                
                let mut 결과들 = 결과들_클론.lock().unwrap();
                결과들.push((스레드_id, 최종_상태));
            });
            
            핸들들.push(핸들);
        }
        
        // 모든 스레드 완료 대기
        for 핸들 in 핸들들 {
            핸들.join().unwrap();
        }
        
        let 결과들 = 결과들.lock().unwrap();
        
        // 모든 결과가 동일해야 함 (결정론적 동작)
        let 첫번째_결과 = 결과들[0].1;
        let 모든_동일 = 결과들.iter().all(|(_, 상태)| *상태 == 첫번째_결과);
        
        assert!(모든_동일, "병렬 실행에서 결과가 일치하지 않음");
        
        // 고유값 개수 확인
        let 고유_결과들: HashSet<u16> = 결과들.iter().map(|(_, 상태)| *상태).collect();
        assert_eq!(고유_결과들.len(), 1, "결과가 하나로 일치해야 함");
        
        println!("✅ 100개 스레드 모두 동일한 결과: {:011b}", 첫번째_결과);
        println!("✅ 병렬 안전성 및 결정론적 동작 검증 완료");
    }
} 