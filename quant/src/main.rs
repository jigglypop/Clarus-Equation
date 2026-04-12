mod ce;
mod data;
mod market;
mod strategy;

use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    println!("=== Clarus Profit-First Quant ===\n");

    let ce = ce::CeConst::derive();
    ce.print_summary();
    println!();

    let bars = if args.len() > 1 {
        let path = PathBuf::from(&args[1]);
        println!("Loading data: {}\n", path.display());
        match data::load_csv(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("CSV load error: {e}");
                std::process::exit(1);
            }
        }
    } else {
        println!("No CSV provided, using synthetic data (1000 bars)\n");
        data::generate_synthetic(1000, 42)
    };

    println!("Bars loaded: {}", bars.len());
    if let (Some(first), Some(last)) = (bars.first(), bars.last()) {
        println!("Period: {} ~ {}", first.date, last.date);
        println!("Price: {:.2} -> {:.2}\n", first.close, last.close);
    }

    let bh = strategy::buy_and_hold(&bars);
    let trend = strategy::stability_trend(&bars);
    let mean_revert = strategy::stability_mean_revert(&bars);
    let walkforward = strategy::walk_forward_suite(&bars, strategy::WalkForwardConfig::default());

    print_header();
    print_row(&bh, &bh);
    print_row(&trend, &bh);
    print_row(&mean_revert, &bh);
    println!("{}", "-".repeat(130));

    println!("\n=== Profit-First Benchmark ===\n");

    let strategies = [&trend, &mean_revert];
    let mut beats_bh = 0;
    let mut total_sharpe = 0.0;
    let mut total_utility = 0.0;

    for s in &strategies {
        let better_return = s.total_return >= bh.total_return;
        let better_cagr = s.cagr >= bh.cagr;
        let better_capture = benchmark_capture(s, &bh) >= 100.0;
        let better_dd = s.max_drawdown < bh.max_drawdown;
        let within_budget = s.max_drawdown <= 35.0;
        let lower_turnover = s.turnover <= bh.turnover;
        let better_utility = s.utility >= bh.utility;

        if (better_return || better_cagr || better_capture) && within_budget {
            beats_bh += 1;
        }

        total_sharpe += s.sharpe;
        total_utility += s.utility;

        println!("  {} vs BuyAndHold:", s.name);
        println!("    Return: {:+.2}% vs {:+.2}% ({})",
            s.total_return, bh.total_return,
            if better_return { "BETTER" } else { "WORSE" }
        );
        println!("    CAGR:   {:+.2}% vs {:+.2}% ({})",
            s.cagr * 100.0, bh.cagr * 100.0,
            if better_cagr { "BETTER" } else { "WORSE" }
        );
        println!("    Capture: {:.1}% vs 100.0% ({})",
            benchmark_capture(s, &bh),
            if better_capture { "BETTER" } else { "WORSE" }
        );
        println!("    MaxDD:   {:.2}% vs {:.2}% ({})",
            s.max_drawdown, bh.max_drawdown,
            if better_dd { "BETTER" } else { "WORSE" }
        );
        println!("    Turnover: {:.2}x vs {:.2}x ({})",
            s.turnover, bh.turnover,
            if lower_turnover { "BETTER" } else { "WORSE" }
        );
        println!("    Utility: {:.3} vs {:.3} ({})",
            s.utility, bh.utility,
            if better_utility { "BETTER" } else { "WORSE" }
        );
        println!();
    }

    println!("  Summary: {}/{} strategies beat BuyAndHold on profit-first basis within DD budget",
        beats_bh, strategies.len());
    println!("  Avg Sharpe (profit strategies): {:.3}", total_sharpe / strategies.len() as f64);
    println!("  Avg Utility (profit strategies): {:.3}", total_utility / strategies.len() as f64);
    println!("  BuyAndHold CAGR: {:+.2}%", bh.cagr * 100.0);
    println!("  BuyAndHold Utility: {:.3}", bh.utility);

    for s in &strategies {
        if !s.trades.is_empty() {
            println!("\n--- {} Trades ({}) ---", s.name, s.trades.len());
            for (i, t) in s.trades.iter().enumerate().take(20) {
                println!("  #{:3} {} -> {} | {:.2} -> {:.2} | {:+.2}% | {}d | {}",
                    i + 1, t.entry_date, t.exit_date,
                    t.entry_price, t.exit_price, t.pnl_pct, t.holding_days, t.reason);
            }
            if s.trades.len() > 20 {
                println!("  ... and {} more trades", s.trades.len() - 20);
            }
        }
    }

    println!("\n=== Walk-Forward Validation ===\n");
    println!(
        "Config: train {} bars | test {} bars | step {} bars",
        walkforward.config.train_bars,
        walkforward.config.test_bars,
        walkforward.config.step_bars
    );

    if walkforward.reports.is_empty() {
        println!("Not enough bars for walk-forward validation.");
        return;
    }

    let wf_windows = walkforward
        .reports
        .first()
        .map(|report| report.windows.len())
        .unwrap_or(0);
    print_walkforward_header();
    print_walkforward_row("WF BuyAndHold", wf_windows, 0.0, &walkforward.benchmark, &walkforward.benchmark);
    for report in &walkforward.reports {
        print_walkforward_row(
            &report.name,
            report.windows.len(),
            report.avg_train_score,
            &report.aggregate,
            &walkforward.benchmark,
        );
    }
    println!("{}", "-".repeat(142));

    for report in &walkforward.reports {
        let better_return = report.aggregate.total_return >= walkforward.benchmark.total_return;
        let better_cagr = report.aggregate.cagr >= walkforward.benchmark.cagr;
        let better_capture = benchmark_capture(&report.aggregate, &walkforward.benchmark) >= 100.0;
        let better_dd = report.aggregate.max_drawdown < walkforward.benchmark.max_drawdown;
        let lower_turnover = report.aggregate.turnover <= walkforward.benchmark.turnover;
        let better_utility = report.aggregate.utility >= walkforward.benchmark.utility;

        println!("\n  {} vs WF BuyAndHold:", report.name);
        println!(
            "    Return: {:+.2}% vs {:+.2}% ({})",
            report.aggregate.total_return,
            walkforward.benchmark.total_return,
            if better_return { "BETTER" } else { "WORSE" }
        );
        println!(
            "    CAGR:   {:+.2}% vs {:+.2}% ({})",
            report.aggregate.cagr * 100.0,
            walkforward.benchmark.cagr * 100.0,
            if better_cagr { "BETTER" } else { "WORSE" }
        );
        println!(
            "    Capture: {:.1}% vs 100.0% ({})",
            benchmark_capture(&report.aggregate, &walkforward.benchmark),
            if better_capture { "BETTER" } else { "WORSE" }
        );
        println!(
            "    MaxDD:  {:.2}% vs {:.2}% ({})",
            report.aggregate.max_drawdown,
            walkforward.benchmark.max_drawdown,
            if better_dd { "BETTER" } else { "WORSE" }
        );
        println!(
            "    Turnover: {:.2}x vs {:.2}x ({})",
            report.aggregate.turnover,
            walkforward.benchmark.turnover,
            if lower_turnover { "BETTER" } else { "WORSE" }
        );
        println!(
            "    Utility: {:.3} vs {:.3} ({})",
            report.aggregate.utility,
            walkforward.benchmark.utility,
            if better_utility { "BETTER" } else { "WORSE" }
        );

        for (idx, window) in report.windows.iter().enumerate() {
            println!(
                "    W{:02} train {}~{} | test {}~{} | {} | train {:.3}/{:.3} | test {:+.2}% | sh {:.3} | dd {:.2}% | whip {:.1}% | turn {:.2}x | trades {}",
                idx + 1,
                window.train_start,
                window.train_end,
                window.test_start,
                window.test_end,
                window.selection,
                window.train_score,
                window.train_utility,
                window.test_return,
                window.test_sharpe,
                window.test_max_drawdown,
                window.test_whipsaw_rate,
                window.test_turnover,
                window.test_trades
            );
        }
    }
}

fn print_header() {
    println!(
        "{:<22} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Strategy",
        "Return%",
        "CAGR%",
        "Cap%",
        "Sharpe",
        "MaxDD%",
        "Utility",
        "Win%",
        "HoldD",
        "Turn",
        "Trades"
    );
    println!("{}", "-".repeat(126));
}

fn print_row(s: &strategy::StrategyResult, benchmark: &strategy::StrategyResult) {
    println!(
        "{:<22} {:>+10.2} {:>8.2} {:>8.1} {:>8.3} {:>8.2} {:>8.3} {:>8.1} {:>8.1} {:>8.2} {:>8}",
        s.name,
        s.total_return,
        s.cagr * 100.0,
        benchmark_capture(s, benchmark),
        s.sharpe,
        s.max_drawdown,
        s.utility,
        s.win_rate,
        s.avg_holding_days,
        s.turnover,
        s.n_trades
    );
}

fn print_walkforward_header() {
    println!(
        "{:<24} {:>7} {:>9} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Strategy",
        "WinN",
        "TrainSc",
        "Return%",
        "CAGR%",
        "Cap%",
        "Sharpe",
        "MaxDD%",
        "Utility",
        "HoldD",
        "Turn",
        "Trades"
    );
    println!("{}", "-".repeat(142));
}

fn print_walkforward_row(
    name: &str,
    windows: usize,
    avg_train_score: f64,
    result: &strategy::StrategyResult,
    benchmark: &strategy::StrategyResult,
) {
    println!(
        "{:<24} {:>7} {:>9.3} {:>+10.2} {:>8.2} {:>8.1} {:>8.3} {:>8.2} {:>8.3} {:>8.1} {:>8.2} {:>8}",
        name,
        windows,
        avg_train_score,
        result.total_return,
        result.cagr * 100.0,
        benchmark_capture(result, benchmark),
        result.sharpe,
        result.max_drawdown,
        result.utility,
        result.avg_holding_days,
        result.turnover,
        result.n_trades
    );
}

fn benchmark_capture(
    result: &strategy::StrategyResult,
    benchmark: &strategy::StrategyResult,
) -> f64 {
    if benchmark.total_return.abs() < 1e-9 {
        0.0
    } else {
        result.total_return / benchmark.total_return * 100.0
    }
}
