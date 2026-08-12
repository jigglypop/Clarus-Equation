//! 기저함수 관련 타입들

/// 기저 함수 타입 (기존 유지)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BasisFunction {
    SinCosh = 0,
    SinSinh = 1,
    CosCosh = 2,
    CosSinh = 3,
    BesselJ = 4,
    BesselI = 5,
    BesselK = 6,
    BesselY = 7,
    TanhSign = 8,
    SechTri = 9,
    ExpSin = 10,
    Morlet = 11,
}

/// 쌍곡 기저 함수 타입 (새로운 1장 설계)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HyperbolicBasisFunction {
    Sinh = 0,     // sinh(αr)
    Cosh = 1,     // cosh(αr)  
    Tanh = 2,     // tanh(αr)
    Sech2 = 3,    // sech²(αr)
} 