use cfr::external_sampling::ExternalSamplingTrainer;
use cfr::kuhn_game::KuhnGame;

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
    let workers = parse_usize_arg(&args, 2, 1);
    let seed = parse_u64_arg(&args, 3, 21);

    let game = KuhnGame;
    let mut trainer = ExternalSamplingTrainer::new(2);
    if workers <= 1 {
        trainer.train_serial(&game, iterations, seed);
    } else {
        trainer.train_parallel(&game, iterations, workers, seed);
    }

    let ev = trainer.expected_value_against_average_policy(&game, 0);
    let target = KuhnGame::TARGET_EV_PLAYER0;
    let report = trainer
        .measure_oracle_exploitability_two_player(&game)
        .expect("two-player report expected");

    println!("iterations={iterations}");
    println!("workers={workers}");
    println!("seed={seed}");
    println!("player0_ev={ev:.6}");
    println!("target_ev={target:.6}");
    println!("abs_error={:.6}", (ev - target).abs());
    println!("oracle_exploitability={:.6}", report.exploitability);
    println!("infosets={}", trainer.infoset_count());
}
