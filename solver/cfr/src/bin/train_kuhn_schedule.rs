use cfr::external_sampling::ExternalSamplingTrainer;
use cfr::kuhn_game::KuhnGame;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug)]
struct Config {
    total_iterations: usize,
    batch_iterations: usize,
    workers: usize,
    seed: u64,
    checkpoint_every_batches: usize,
    metrics_every_batches: usize,
    checkpoint_dir: PathBuf,
    resume: Option<PathBuf>,
    metrics_out: PathBuf,
}

#[derive(Clone, Debug, Serialize)]
struct MetricPoint {
    unix_seconds: u64,
    iterations: u64,
    player0_ev: f64,
    abs_error: f64,
    exploitability: f64,
    infosets: usize,
}

#[derive(Debug, Serialize)]
struct MetricsDocument {
    total_iterations: usize,
    batch_iterations: usize,
    workers: usize,
    seed: u64,
    checkpoint_every_batches: usize,
    metrics_every_batches: usize,
    checkpoint_dir: String,
    resume: Option<String>,
    points: Vec<MetricPoint>,
}

fn parse_usize_arg(args: &[String], name: &str, default: usize) -> usize {
    flag_value(args, name)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_u64_arg(args: &[String], name: &str, default: u64) -> u64 {
    flag_value(args, name)
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

fn parse_path_arg(args: &[String], name: &str, default: &str) -> PathBuf {
    PathBuf::from(flag_value(args, name).unwrap_or(default))
}

fn parse_optional_path_arg(args: &[String], name: &str) -> Option<PathBuf> {
    flag_value(args, name).map(PathBuf::from)
}

fn flag_value<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.windows(2).find_map(|w| {
        if w[0] == name {
            Some(w[1].as_str())
        } else {
            None
        }
    })
}

fn parse_config(args: &[String]) -> Config {
    Config {
        total_iterations: parse_usize_arg(args, "--total-iterations", 1_000_000),
        batch_iterations: parse_usize_arg(args, "--batch-iterations", 100_000),
        workers: parse_usize_arg(args, "--workers", 4),
        seed: parse_u64_arg(args, "--seed", 21),
        checkpoint_every_batches: parse_usize_arg(args, "--checkpoint-every-batches", 1).max(1),
        metrics_every_batches: parse_usize_arg(args, "--metrics-every-batches", 1).max(1),
        checkpoint_dir: parse_path_arg(args, "--checkpoint-dir", "checkpoints/kuhn_linear"),
        resume: parse_optional_path_arg(args, "--resume"),
        metrics_out: parse_path_arg(args, "--metrics-out", "data/kuhn_linear_train_metrics.json"),
    }
}

fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn write_metrics(path: &Path, doc: &MetricsDocument) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("failed to create metrics parent directory");
    }
    let body = serde_json::to_string_pretty(doc).expect("failed to serialize metrics");
    fs::write(path, body).expect("failed to write metrics file");
}

fn save_checkpoint(trainer: &ExternalSamplingTrainer, checkpoint_dir: &Path) {
    fs::create_dir_all(checkpoint_dir).expect("failed to create checkpoint directory");
    let iter_file = checkpoint_dir.join(format!("iter_{:012}.ckpt", trainer.iterations));
    trainer
        .save_checkpoint(&iter_file)
        .expect("failed to save iteration checkpoint");
    let latest_file = checkpoint_dir.join("latest.ckpt");
    trainer
        .save_checkpoint(&latest_file)
        .expect("failed to save latest checkpoint");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!(
            "train_kuhn_schedule options:\n\
             --total-iterations <usize> (default 1000000)\n\
             --batch-iterations <usize> (default 100000)\n\
             --workers <usize> (default 4)\n\
             --seed <u64> (default 21)\n\
             --checkpoint-every-batches <usize> (default 1)\n\
             --metrics-every-batches <usize> (default 1)\n\
             --checkpoint-dir <path> (default checkpoints/kuhn_linear)\n\
             --metrics-out <path> (default data/kuhn_linear_train_metrics.json)\n\
             --resume <path> (optional)"
        );
        return;
    }

    let config = parse_config(&args);
    let game = KuhnGame;

    let mut trainer = if let Some(path) = &config.resume {
        if path.exists() {
            ExternalSamplingTrainer::load_checkpoint(path)
                .unwrap_or_else(|err| panic!("failed to load checkpoint {}: {err}", path.display()))
        } else {
            ExternalSamplingTrainer::new(2)
        }
    } else {
        ExternalSamplingTrainer::new(2)
    };

    let target_total = config.total_iterations as u64;
    if trainer.iterations > target_total {
        println!(
            "checkpoint iterations ({}) already exceed target iterations ({})",
            trainer.iterations, target_total
        );
        return;
    }

    let mut points = Vec::new();
    let mut batch_idx = 0usize;
    while trainer.iterations < target_total {
        let remaining = (target_total - trainer.iterations) as usize;
        let batch = remaining.min(config.batch_iterations);
        let batch_seed = config.seed
            ^ ((trainer.iterations + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            ^ (batch_idx as u64);

        if config.workers <= 1 {
            trainer.train_serial(&game, batch, batch_seed);
        } else {
            trainer.train_parallel(&game, batch, config.workers, batch_seed);
        }

        batch_idx += 1;
        let done = trainer.iterations >= target_total;

        if batch_idx % config.metrics_every_batches == 0 || done {
            let ev = trainer.expected_value_against_average_policy(&game, 0);
            let report = trainer
                .measure_oracle_exploitability_two_player(&game)
                .expect("two-player exploitability report");
            let point = MetricPoint {
                unix_seconds: now_unix_seconds(),
                iterations: trainer.iterations,
                player0_ev: ev,
                abs_error: (ev - KuhnGame::TARGET_EV_PLAYER0).abs(),
                exploitability: report.exploitability,
                infosets: trainer.infoset_count(),
            };
            println!(
                "iter={} ev={:.6} abs_error={:.6} oracle_exploitability={:.6} infosets={}",
                point.iterations,
                point.player0_ev,
                point.abs_error,
                point.exploitability,
                point.infosets
            );
            points.push(point);
            let doc = MetricsDocument {
                total_iterations: config.total_iterations,
                batch_iterations: config.batch_iterations,
                workers: config.workers,
                seed: config.seed,
                checkpoint_every_batches: config.checkpoint_every_batches,
                metrics_every_batches: config.metrics_every_batches,
                checkpoint_dir: config.checkpoint_dir.display().to_string(),
                resume: config.resume.as_ref().map(|p| p.display().to_string()),
                points: points.clone(),
            };
            write_metrics(&config.metrics_out, &doc);
        }

        if batch_idx % config.checkpoint_every_batches == 0 || done {
            save_checkpoint(&trainer, &config.checkpoint_dir);
        }
    }

    println!(
        "completed total_iterations={} final_infosets={}",
        trainer.iterations,
        trainer.infoset_count()
    );
}
