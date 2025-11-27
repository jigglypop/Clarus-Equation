use std::collections::VecDeque;

/// SFE 시장 기하학 엔진
pub struct MarketGeometry {
    window: usize,
    alpha: f64,
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    high_history: VecDeque<f64>,
    low_history: VecDeque<f64>,
}

impl MarketGeometry {
    pub fn new(window: usize, alpha: f64) -> Self {
        Self {
            window,
            alpha,
            price_history: VecDeque::with_capacity(window),
            volume_history: VecDeque::with_capacity(window),
            high_history: VecDeque::with_capacity(window),
            low_history: VecDeque::with_capacity(window),
        }
    }

    /// 새로운 틱 데이터 업데이트 및 곡률 계산
    pub fn update(&mut self, price: f64, volume: f64, high: f64, low: f64) -> (f64, f64) {
        if self.price_history.len() >= self.window {
            self.price_history.pop_front();
            self.volume_history.pop_front();
            self.high_history.pop_front();
            self.low_history.pop_front();
        }
        
        self.price_history.push_back(price);
        self.volume_history.push_back(volume);
        self.high_history.push_back(high);
        self.low_history.push_back(low);
        
        if self.price_history.len() < self.window {
            return (0.0, 1.0); // 데이터 부족 시 기본값 (R=0, Phi=1)
        }
        
        self.calculate_curvature()
    }
    
    /// 시장 곡률(R)과 억압장(Phi) 계산
    fn calculate_curvature(&self) -> (f64, f64) {
        // 1. Volatility (Parkinson Estimator approximation using history)
        let mut sum_hl_sq = 0.0;
        for (h, l) in self.high_history.iter().zip(self.low_history.iter()) {
            let hl_ratio = (h / l).ln();
            sum_hl_sq += hl_ratio * hl_ratio;
        }
        let volatility = (sum_hl_sq / (4.0 * 2.0_f64.ln() * self.window as f64)).sqrt();
        
        // 2. Relative Liquidity
        let avg_vol: f64 = self.volume_history.iter().sum::<f64>() / self.window as f64;
        let current_vol = *self.volume_history.back().unwrap_or(&1.0);
        let relative_liquidity = current_vol / (avg_vol + 1e-9);
        
        // 3. Curvature R
        // R ~ Vol^2 / Liq
        let r = self.alpha * (volatility.powi(2)) / (relative_liquidity + 0.1);
        
        // 4. Suppression Phi
        let phi = (-r * 100.0).exp(); // Scale factor
        
        (r, phi)
    }
}

/// SFE 트레이딩 봇 (SFE Surfing Strategy)
pub struct SfeTradingBot {
    engine: MarketGeometry,
    position: f64, // 1.0 (Long), 0.0 (None), -1.0 (Short)
    entry_price: f64,
    cash: f64,
    holdings: f64,
    
    // 전략 파라미터
    r_threshold_entry: f64, // 진입 임계치 (위기 감지)
    r_peak: f64,            // 곡률 피크 추적용
    in_crisis: bool,        // 위기 상황 플래그
}

impl SfeTradingBot {
    pub fn new(initial_cash: f64) -> Self {
        Self {
            engine: MarketGeometry::new(20, 5.0),
            position: 0.0,
            entry_price: 0.0,
            cash: initial_cash,
            holdings: 0.0,
            
            r_threshold_entry: 0.0001, // 임계치 대폭 하향 (민감도 증가)
            r_peak: 0.0,
            in_crisis: false,
        }
    }
    
    pub fn process_tick(&mut self, price: f64, volume: f64, high: f64, low: f64) -> f64 {
        let (r, phi) = self.engine.update(price, volume, high, low);
        
        // 전략 로직: "Buy the Fear (High Curvature Peak), Sell the Greed (Low Curvature)"
        
        if self.position == 0.0 {
            // [진입 로직]
            if r > self.r_threshold_entry {
                self.in_crisis = true;
                if r > self.r_peak {
                    self.r_peak = r; // 피크 갱신 중 (아직 바닥 아님)
                } else if r < self.r_peak * 0.9 { // 10% 진정 시 진입 (더 빠른 진입)
                    self.buy(price);
                    self.in_crisis = false;
                    self.r_peak = 0.0;
                }
            } else {
                self.r_peak = 0.0;
            }
        } else {
            // [청산 로직]
            // 시장이 안정화(Phi > 0.99)되거나, 손절매(-5%)
            if phi > 0.995 { // 더 엄격한 안정화 기준
                self.sell(price, "Stabilized");
            } else if price < self.entry_price * 0.90 { // 손절 폭 10%로 확대
                self.sell(price, "StopLoss");
            }
        }
        
        self.get_portfolio_value(price)
    }
    
    fn buy(&mut self, price: f64) {
        let amount = self.cash / price;
        self.holdings = amount;
        self.cash = 0.0;
        self.position = 1.0;
        self.entry_price = price;
        println!("  [BUY] Price: {:.2}", price);
    }
    
    fn sell(&mut self, price: f64, reason: &str) {
        let amount = self.holdings * price;
        self.cash = amount;
        self.holdings = 0.0;
        self.position = 0.0;
        println!("  [SELL] Price: {:.2} ({})", price, reason);
    }
    
    pub fn get_portfolio_value(&self, current_price: f64) -> f64 {
        self.cash + self.holdings * current_price
    }
}

/// [NEW] 거시경제 위기 예측 시스템
pub struct MacroEconomy {
    // 상태 변수
    pub gdp_growth: f64,      // % (e.g., 0.03 for 3%)
    pub debt_to_gdp: f64,     // ratio (e.g., 1.0 for 100%)
    pub interest_rate: f64,   // % (e.g., 0.05)
    pub m2_growth: f64,       // %
    
    // SFE 파라미터
    alpha_debt: f64,          // 부채 민감도
    beta_liquidity: f64,      // 유동성 민감도
}

impl MacroEconomy {
    pub fn new() -> Self {
        Self {
            gdp_growth: 0.03,
            debt_to_gdp: 0.8,
            interest_rate: 0.03,
            m2_growth: 0.05,
            alpha_debt: 10.0, // 민감도 상향 (2.0 -> 10.0)
            beta_liquidity: 1.5,
        }
    }

    /// 경제 상태 업데이트 및 위기 확률 계산
    /// 반환값: (Macro_Curvature_R, Crisis_Probability)
    pub fn update_and_predict(&mut self, 
        gdp_shock: f64, debt_shock: f64, rate_shock: f64, m2_shock: f64
    ) -> (f64, f64) {
        // 1. 상태 진화 (충격 반영)
        self.gdp_growth += gdp_shock;
        self.debt_to_gdp += debt_shock;
        self.interest_rate += rate_shock;
        self.m2_growth += m2_shock;

        // 2. SFE 거시 곡률(R) 계산
        // 이론: R ~ (부채압력 / 성장동력) * 금리부담
        
        // 부채 압력: (Debt/GDP)^2
        let debt_pressure = self.debt_to_gdp.powi(2);
        
        // 성장 동력: GDP성장 + M2증가 (유동성 함정 반영)
        // GDP가 마이너스면 유동성(M2) 효과가 급감함 (돈맥경화)
        let liquidity_efficiency = if self.gdp_growth < 0.0 { 0.1 } else { 1.0 };
        let growth_power = (self.gdp_growth + self.m2_growth * 0.5 * liquidity_efficiency).max(0.0001); 
        
        // 금리 부담: e^(Interest Rate)
        let rate_burden = (self.interest_rate * 20.0).exp(); // 금리 민감도 상향

        // 최종 거시 곡률
        let r_macro = self.alpha_debt * (debt_pressure / growth_power) * rate_burden * 0.001;
        
        // 3. 위기 확률 (Crisis Probability)
        let prob_crisis = 1.0 - (-r_macro).exp();
        
        (r_macro, prob_crisis)
    }
}
