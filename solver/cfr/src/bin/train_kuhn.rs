use cfr::kuhn::KuhnTrainer;

fn parse_usize_arg(args: &[String], idx: usize, default: usize) -> usize {
    args.get(idx)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_u64_arg(args: &[String], idx: usize, default: u64) -> u64 {
    args.get(idx)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iterations = parse_usize_arg(&args, 1, 1_000_000);
    let seed = parse_u64_arg(&args, 2, 17);

    let mut trainer = KuhnTrainer::new();
    trainer.train(iterations, seed);
    let ev = trainer.expected_value_player0();
    let target = -1.0 / 18.0;

    println!("iterations={iterations}");
    println!("seed={seed}");
    println!("player0_ev={ev:.6}");
    println!("target_ev={target:.6}");
    println!("abs_error={:.6}", (ev - target).abs());
}
