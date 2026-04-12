use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Bar {
    pub date: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Deserialize)]
struct CsvRow {
    #[serde(alias = "Date", alias = "date", alias = "datetime")]
    date: String,
    #[serde(alias = "Open", alias = "open")]
    open: f64,
    #[serde(alias = "High", alias = "high")]
    high: f64,
    #[serde(alias = "Low", alias = "low")]
    low: f64,
    #[serde(alias = "Close", alias = "close", alias = "Adj Close")]
    close: f64,
    #[serde(alias = "Volume", alias = "volume")]
    volume: f64,
}

pub fn load_csv(path: &Path) -> Result<Vec<Bar>, Box<dyn std::error::Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut bars = Vec::new();
    for result in rdr.deserialize() {
        let row: CsvRow = result?;
        if row.high <= 0.0 || row.low <= 0.0 || row.close <= 0.0 {
            continue;
        }
        bars.push(Bar {
            date: row.date,
            open: row.open,
            high: row.high,
            low: row.low,
            close: row.close,
            volume: row.volume,
        });
    }
    Ok(bars)
}

pub fn generate_synthetic(n: usize, seed: u64) -> Vec<Bar> {
    let mut bars = Vec::with_capacity(n);
    let mut price = 100.0;
    let mut rng = LcgRng::new(seed);

    for i in 0..n {
        let ret = (rng.next_f64() - 0.5) * 0.04 + 0.0002;
        let vol_factor = 1.0 + (rng.next_f64() - 0.5) * 0.03;
        price *= 1.0 + ret;

        let high = price * (1.0 + rng.next_f64() * 0.015);
        let low = price * (1.0 - rng.next_f64() * 0.015);
        let volume = 1_000_000.0 * vol_factor;

        if i > 200 && i < 230 {
            price *= 0.985;
        }
        if i > 500 && i < 520 {
            price *= 0.99;
        }

        bars.push(Bar {
            date: format!("2020-{:02}-{:02}", (i / 30) % 12 + 1, i % 28 + 1),
            open: price * 0.999,
            high: high.max(price),
            low: low.min(price),
            close: price,
            volume,
        });
    }
    bars
}

struct LcgRng { state: u64 }
impl LcgRng {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}
