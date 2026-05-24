use cfr::external_sampling::ExternalSamplingTrainer;
use cfr::nlhe_game::NlheGameModel;
use game::NlheConfig;
use serde::Serialize;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug)]
struct Config {
    total_iterations: usize,
    batch_iterations: usize,
    sub_batch_iterations: usize,
    max_runtime_seconds: Option<u64>,
    workers: usize,
    seed: u64,
    deck_samples: usize,
    checkpoint_every_batches: usize,
    metrics_every_batches: usize,
    skip_exploitability: bool,
    exploitability_every_batches: usize,
    exploitability_samples: usize,
    checkpoint_dir: PathBuf,
    resume: Option<PathBuf>,
    metrics_out: PathBuf,
    cluster_dir: PathBuf,
    num_players: usize,
    starting_stack: u32,
    small_blind: u32,
    big_blind: u32,
}

#[derive(Clone, Debug, Serialize)]
struct MetricPoint {
    unix_seconds: u64,
    iterations: u64,
    expected_values: Vec<f64>,
    exploitability_two_player: Option<f64>,
    sampled_exploitability_two_player: Option<f64>,
    oracle_exploitability_two_player: Option<f64>,
    infoset_exploitability_two_player: Option<f64>,
    infosets: usize,
    non_uniform_infosets: usize,
    non_uniform_pct: f64,
}

#[derive(Debug, Serialize)]
struct MetricsDocument {
    total_iterations: usize,
    batch_iterations: usize,
    sub_batch_iterations: usize,
    max_runtime_seconds: Option<u64>,
    workers: usize,
    seed: u64,
    deck_samples: usize,
    checkpoint_every_batches: usize,
    metrics_every_batches: usize,
    skip_exploitability: bool,
    exploitability_every_batches: usize,
    exploitability_samples: usize,
    checkpoint_dir: String,
    resume: Option<String>,
    metrics_out: String,
    cluster_dir: String,
    num_players: usize,
    starting_stack: u32,
    small_blind: u32,
    big_blind: u32,
    points: Vec<MetricPoint>,
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

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
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

fn parse_u32_arg(args: &[String], name: &str, default: u32) -> u32 {
    flag_value(args, name)
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(default)
}

fn parse_path_arg(args: &[String], name: &str, default: &str) -> PathBuf {
    PathBuf::from(flag_value(args, name).unwrap_or(default))
}

fn parse_optional_path_arg(args: &[String], name: &str) -> Option<PathBuf> {
    flag_value(args, name).map(PathBuf::from)
}

fn default_workers() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn parse_config(args: &[String]) -> Config {
    let max_runtime_seconds = parse_usize_arg(args, "--max-runtime-seconds", 0);
    Config {
        total_iterations: parse_usize_arg(args, "--total-iterations", 2_000_000),
        batch_iterations: parse_usize_arg(args, "--batch-iterations", 100_000),
        sub_batch_iterations: parse_usize_arg(args, "--sub-batch-iterations", 10_000).max(1),
        max_runtime_seconds: (max_runtime_seconds > 0).then_some(max_runtime_seconds as u64),
        workers: parse_usize_arg(args, "--workers", default_workers()).max(1),
        seed: parse_u64_arg(args, "--seed", 21),
        deck_samples: parse_usize_arg(args, "--deck-samples", 1000).max(1),
        checkpoint_every_batches: parse_usize_arg(args, "--checkpoint-every-batches", 1).max(1),
        metrics_every_batches: parse_usize_arg(args, "--metrics-every-batches", 1).max(1),
        skip_exploitability: has_flag(args, "--skip-exploitability"),
        exploitability_every_batches: parse_usize_arg(args, "--exploitability-every-batches", 5)
            .max(1),
        exploitability_samples: parse_usize_arg(args, "--exploitability-samples", 50).max(1),
        checkpoint_dir: parse_path_arg(args, "--checkpoint-dir", "checkpoints/nlhe_hu"),
        resume: parse_optional_path_arg(args, "--resume"),
        metrics_out: parse_path_arg(args, "--metrics-out", "data/nlhe_train_metrics.json"),
        cluster_dir: parse_path_arg(args, "--cluster-dir", "checkpoints/nlhe_clusters"),
        num_players: parse_usize_arg(args, "--num-players", 2).max(2),
        starting_stack: parse_u32_arg(args, "--starting-stack", 2_000),
        small_blind: parse_u32_arg(args, "--small-blind", 10),
        big_blind: parse_u32_arg(args, "--big-blind", 20),
    }
}

fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn flush_stdout() {
    let _ = std::io::stdout().flush();
}

fn ceil_div_usize(value: usize, divisor: usize) -> usize {
    if value == 0 {
        0
    } else {
        1 + (value - 1) / divisor.max(1)
    }
}

fn format_duration(seconds: f64) -> String {
    let total = seconds.max(0.0).round() as u64;
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let secs = total % 60;
    if hours > 0 {
        format!("{hours}h{minutes:02}m{secs:02}s")
    } else if minutes > 0 {
        format!("{minutes}m{secs:02}s")
    } else {
        format!("{secs}s")
    }
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

fn build_deck_seeds(deck_samples: usize, seed: u64) -> Vec<u64> {
    (0..deck_samples)
        .map(|idx| seed ^ ((idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)))
        .collect()
}

fn print_help() {
    println!(
        "train_nlhe options:\n\
         --total-iterations <usize> (default 2000000)\n\
         --batch-iterations <usize> (default 100000)\n\
         --sub-batch-iterations <usize> (default 10000)\n\
         --max-runtime-seconds <usize> (default 0 = no timeout)\n\
         --workers <usize> (default all logical cores)\n\
         --seed <u64> (default 21)\n\
         --deck-samples <usize> (default 1000)\n\
         --checkpoint-every-batches <usize> (default 1)\n\
         --metrics-every-batches <usize> (default 1)\n\
         --skip-exploitability (optional)\n\
         --exploitability-every-batches <usize> (default 5)\n\
         --exploitability-samples <usize> (default 50)\n\
         --checkpoint-dir <path> (default checkpoints/nlhe_hu)\n\
         --metrics-out <path> (default data/nlhe_train_metrics.json)\n\
         --cluster-dir <path> (default checkpoints/nlhe_clusters)\n\
         --num-players <usize> (default 2)\n\
         --starting-stack <u32> (default 2000)\n\
         --small-blind <u32> (default 10)\n\
         --big-blind <u32> (default 20)\n\
         --resume <path> (optional)"
    );
}

fn print_startup_banner(config: &Config, starting_iteration: u64, target_total: u64) {
    let remaining_iterations = target_total.saturating_sub(starting_iteration) as usize;
    let remaining_batches = ceil_div_usize(remaining_iterations, config.batch_iterations);
    let sub_batches_per_batch =
        ceil_div_usize(config.batch_iterations, config.sub_batch_iterations);

    println!("=== MCCFR NLHE Training ===");
    println!("  target: {} iterations", target_total);
    println!(
        "  current: {} (remaining {})",
        starting_iteration, remaining_iterations
    );
    println!("  workers: {} threads", config.workers);
    println!(
        "  batch: {} ({} sub-batches of {})",
        config.batch_iterations, sub_batches_per_batch, config.sub_batch_iterations
    );
    if let Some(limit) = config.max_runtime_seconds {
        println!("  timeout: {} (wall clock)", format_duration(limit as f64));
    } else {
        println!("  timeout: none");
    }
    println!("  remaining batches: {}", remaining_batches);
    println!("  deck samples: {}", config.deck_samples);
    println!("  clusters: {}", config.cluster_dir.display());
    println!(
        "  checkpoint: {} (every {} batches)",
        config.checkpoint_dir.display(),
        config.checkpoint_every_batches
    );
    println!(
        "  metrics: {} (every {} batches)",
        config.metrics_out.display(),
        config.metrics_every_batches
    );
    if config.skip_exploitability {
        println!("  exploitability: skipped");
    } else {
        println!(
            "  exploitability checks: every {} batches",
            config.exploitability_every_batches
        );
        println!(
            "  primary: sampled infoset exploitability ({} root samples)",
            config.exploitability_samples
        );
    }
    println!(
        "  resume: {}",
        config
            .resume
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "(none)".to_string())
    );
    println!("===========================");
    flush_stdout();
}

fn train_main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_help();
        return;
    }

    let config = parse_config(&args);
    let deck_seeds = build_deck_seeds(config.deck_samples, config.seed);
    let game_model = NlheGameModel::new(
        NlheConfig {
            num_players: config.num_players,
            starting_stack: config.starting_stack,
            small_blind: config.small_blind,
            big_blind: config.big_blind,
        },
        deck_seeds,
        &config.cluster_dir,
    )
    .unwrap_or_else(|err| {
        panic!(
            "failed to construct NLHE model with cluster_dir={}: {err}",
            config.cluster_dir.display()
        )
    });

    let mut trainer = if let Some(path) = &config.resume {
        if path.exists() {
            ExternalSamplingTrainer::load_checkpoint(path)
                .unwrap_or_else(|err| panic!("failed to load checkpoint {}: {err}", path.display()))
        } else {
            ExternalSamplingTrainer::new(config.num_players)
        }
    } else {
        ExternalSamplingTrainer::new(config.num_players)
    };

    if trainer.num_players != config.num_players {
        panic!(
            "checkpoint player count mismatch: checkpoint has {}, requested {}",
            trainer.num_players, config.num_players
        );
    }

    let target_total = config.total_iterations as u64;
    if trainer.iterations > target_total {
        println!(
            "checkpoint iterations ({}) already exceed target iterations ({})",
            trainer.iterations, target_total
        );
        flush_stdout();
        return;
    }

    print_startup_banner(&config, trainer.iterations, target_total);
    if config.max_runtime_seconds.is_some() && !config.skip_exploitability {
        println!("[timeout mode] disabling exploitability checks to honor wall-clock budget");
        flush_stdout();
    }

    let mut points = Vec::new();
    let mut batch_idx = 0usize;
    let mut last_expected_values = Vec::<f64>::new();
    let mut last_sampled_exploitability_two_player = None::<f64>;
    let started_at = Instant::now();
    let starting_iterations = trainer.iterations;
    let mut timed_out = false;
    while trainer.iterations < target_total && !timed_out {
        if let Some(limit_s) = config.max_runtime_seconds {
            let elapsed_s = started_at.elapsed().as_secs();
            if elapsed_s >= limit_s {
                timed_out = true;
                println!(
                    "[timeout] reached wall-clock limit {} at iter={}",
                    format_duration(limit_s as f64),
                    trainer.iterations
                );
                flush_stdout();
                break;
            }
        }

        let remaining = (target_total - trainer.iterations) as usize;
        let batch = remaining.min(config.batch_iterations);
        let batch_seed = config.seed
            ^ ((trainer.iterations + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            ^ (batch_idx as u64);
        batch_idx += 1;

        let mut done_in_batch = 0usize;
        let sub_batch_size = config.sub_batch_iterations.min(batch).max(1);
        let total_sub_batches = ceil_div_usize(batch, sub_batch_size);
        while done_in_batch < batch {
            if let Some(limit_s) = config.max_runtime_seconds {
                let elapsed_s = started_at.elapsed().as_secs();
                if elapsed_s >= limit_s {
                    timed_out = true;
                    println!(
                        "[timeout] reached wall-clock limit {} during batch {} at iter={}",
                        format_duration(limit_s as f64),
                        batch_idx,
                        trainer.iterations
                    );
                    flush_stdout();
                    break;
                }
            }

            let sub_idx = (done_in_batch / sub_batch_size) + 1;
            let sub_batch = (batch - done_in_batch).min(sub_batch_size);
            let sub_seed = batch_seed
                ^ ((sub_idx as u64).wrapping_mul(0xD1B5_4A32_D192_ED03))
                ^ (trainer.iterations.wrapping_mul(0x94D0_49BB_1331_11EB));
            let sub_started_at = Instant::now();

            if config.workers <= 1 {
                trainer.train_serial(&game_model, sub_batch, sub_seed);
            } else {
                trainer.train_parallel(&game_model, sub_batch, config.workers, sub_seed);
            }
            done_in_batch += sub_batch;

            let sub_elapsed = sub_started_at.elapsed().as_secs_f64().max(1e-9);
            let sub_rate = sub_batch as f64 / sub_elapsed;
            let total_elapsed = started_at.elapsed().as_secs_f64().max(1e-9);
            let completed = trainer.iterations.saturating_sub(starting_iterations);
            let avg_rate = completed as f64 / total_elapsed;
            let remaining_total = target_total.saturating_sub(trainer.iterations);
            let eta_seconds = if avg_rate > 0.0 {
                remaining_total as f64 / avg_rate
            } else {
                0.0
            };

            println!(
                "[batch {batch_idx}] sub {sub_idx}/{total_sub_batches}: +{sub_batch} iters in {:.2}s ({:.0} iter/s) | total={}/{} | infosets={} | elapsed={} | ETA ~{}",
                sub_elapsed,
                sub_rate,
                trainer.iterations,
                target_total,
                trainer.infoset_count(),
                format_duration(total_elapsed),
                format_duration(eta_seconds)
            );
            flush_stdout();
        }

        let reached_target = trainer.iterations >= target_total;
        let done = reached_target || timed_out;
        let should_compute_exploitability = !config.skip_exploitability
            && config.max_runtime_seconds.is_none()
            && !timed_out
            && (batch_idx % config.exploitability_every_batches == 0 || reached_target);

        if should_compute_exploitability {
            println!(
                "[exploitability check] computing sampled infoset exploitability (samples={})...",
                config.exploitability_samples
            );
            flush_stdout();
            let exploitability_started_at = Instant::now();
            let sample_seed = batch_seed
                ^ (trainer.iterations.wrapping_mul(0xA24B_AED4_0B7F_4D95))
                ^ ((batch_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let sampled_report = trainer.measure_sampled_exploitability_two_player(
                &game_model,
                config.exploitability_samples,
                sample_seed,
            );
            if let Some(ref report) = sampled_report {
                last_expected_values = report.expected_values.clone();
            }
            last_sampled_exploitability_two_player = sampled_report.map(|r| r.exploitability);
            let exploitability_elapsed = exploitability_started_at.elapsed().as_secs_f64();
            let player0_ev = last_expected_values.first().copied().unwrap_or(0.0);
            println!(
                "[exploitability check] done: sampled_infoset_exploitability={:?} player0_ev={:.6} (took {:.1}s)",
                last_sampled_exploitability_two_player,
                player0_ev,
                exploitability_elapsed
            );
            flush_stdout();
        }

        if batch_idx % config.metrics_every_batches == 0 || done {
            let (total_infosets, non_uniform_infosets, _avg_actions, _max_actions) =
                trainer.diagnostic_info();
            let non_uniform_pct = if total_infosets > 0 {
                non_uniform_infosets as f64 / total_infosets as f64 * 100.0
            } else {
                0.0
            };
            let point = MetricPoint {
                unix_seconds: now_unix_seconds(),
                iterations: trainer.iterations,
                expected_values: last_expected_values.clone(),
                exploitability_two_player: last_sampled_exploitability_two_player,
                sampled_exploitability_two_player: last_sampled_exploitability_two_player,
                oracle_exploitability_two_player: None,
                infoset_exploitability_two_player: last_sampled_exploitability_two_player,
                infosets: total_infosets,
                non_uniform_infosets,
                non_uniform_pct,
            };

            let player0_ev = point
                .expected_values
                .first()
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "n/a".to_string());
            println!(
                "[metrics] iter={} player0_ev={} sampled_infoset_exploitability={:?} infosets={} non_uniform={}/{} ({:.1}%)",
                point.iterations,
                player0_ev,
                point.sampled_exploitability_two_player,
                point.infosets,
                point.non_uniform_infosets,
                point.infosets,
                point.non_uniform_pct
            );
            flush_stdout();
            points.push(point);

            let doc = MetricsDocument {
                total_iterations: config.total_iterations,
                batch_iterations: config.batch_iterations,
                sub_batch_iterations: config.sub_batch_iterations,
                max_runtime_seconds: config.max_runtime_seconds,
                workers: config.workers,
                seed: config.seed,
                deck_samples: config.deck_samples,
                checkpoint_every_batches: config.checkpoint_every_batches,
                metrics_every_batches: config.metrics_every_batches,
                skip_exploitability: config.skip_exploitability,
                exploitability_every_batches: config.exploitability_every_batches,
                exploitability_samples: config.exploitability_samples,
                checkpoint_dir: config.checkpoint_dir.display().to_string(),
                resume: config.resume.as_ref().map(|p| p.display().to_string()),
                metrics_out: config.metrics_out.display().to_string(),
                cluster_dir: config.cluster_dir.display().to_string(),
                num_players: config.num_players,
                starting_stack: config.starting_stack,
                small_blind: config.small_blind,
                big_blind: config.big_blind,
                points: points.clone(),
            };
            write_metrics(&config.metrics_out, &doc);
        }

        if batch_idx % config.checkpoint_every_batches == 0 || done {
            save_checkpoint(&trainer, &config.checkpoint_dir);
            println!(
                "[checkpoint] saved iter={} to {}",
                trainer.iterations,
                config.checkpoint_dir.display()
            );
            flush_stdout();
        }

        if timed_out {
            break;
        }
    }

    let needs_final_metrics_snapshot = points
        .last()
        .map(|point| point.iterations != trainer.iterations)
        .unwrap_or(true);
    if needs_final_metrics_snapshot {
        let should_compute_final_exploitability =
            !config.skip_exploitability && config.max_runtime_seconds.is_none();
        if should_compute_final_exploitability {
            println!(
                "[final exploitability snapshot] computing sampled infoset exploitability (samples={})...",
                config.exploitability_samples
            );
            flush_stdout();
            let final_seed = config.seed ^ (trainer.iterations.wrapping_mul(0xF135_7AEA_2E62_A9C5));
            let sampled_report = trainer.measure_sampled_exploitability_two_player(
                &game_model,
                config.exploitability_samples,
                final_seed,
            );
            if let Some(ref report) = sampled_report {
                last_expected_values = report.expected_values.clone();
            }
            last_sampled_exploitability_two_player = sampled_report.map(|r| r.exploitability);
            println!(
                "[final exploitability snapshot] sampled_infoset_exploitability={:?}",
                last_sampled_exploitability_two_player
            );
            flush_stdout();
        }

        let (total_infosets, non_uniform_infosets, _avg_actions, _max_actions) =
            trainer.diagnostic_info();
        let non_uniform_pct = if total_infosets > 0 {
            non_uniform_infosets as f64 / total_infosets as f64 * 100.0
        } else {
            0.0
        };
        let point = MetricPoint {
            unix_seconds: now_unix_seconds(),
            iterations: trainer.iterations,
            expected_values: last_expected_values.clone(),
            exploitability_two_player: last_sampled_exploitability_two_player,
            sampled_exploitability_two_player: last_sampled_exploitability_two_player,
            oracle_exploitability_two_player: None,
            infoset_exploitability_two_player: last_sampled_exploitability_two_player,
            infosets: total_infosets,
            non_uniform_infosets,
            non_uniform_pct,
        };
        let player0_ev = point
            .expected_values
            .first()
            .map(|v| format!("{v:.6}"))
            .unwrap_or_else(|| "n/a".to_string());
        println!(
            "[metrics] iter={} player0_ev={} sampled_infoset_exploitability={:?} infosets={} non_uniform={}/{} ({:.1}%)",
            point.iterations,
            player0_ev,
            point.sampled_exploitability_two_player,
            point.infosets,
            point.non_uniform_infosets,
            point.infosets,
            point.non_uniform_pct
        );
        flush_stdout();
        points.push(point);

        let doc = MetricsDocument {
            total_iterations: config.total_iterations,
            batch_iterations: config.batch_iterations,
            sub_batch_iterations: config.sub_batch_iterations,
            max_runtime_seconds: config.max_runtime_seconds,
            workers: config.workers,
            seed: config.seed,
            deck_samples: config.deck_samples,
            checkpoint_every_batches: config.checkpoint_every_batches,
            metrics_every_batches: config.metrics_every_batches,
            skip_exploitability: config.skip_exploitability,
            exploitability_every_batches: config.exploitability_every_batches,
            exploitability_samples: config.exploitability_samples,
            checkpoint_dir: config.checkpoint_dir.display().to_string(),
            resume: config.resume.as_ref().map(|p| p.display().to_string()),
            metrics_out: config.metrics_out.display().to_string(),
            cluster_dir: config.cluster_dir.display().to_string(),
            num_players: config.num_players,
            starting_stack: config.starting_stack,
            small_blind: config.small_blind,
            big_blind: config.big_blind,
            points: points.clone(),
        };
        write_metrics(&config.metrics_out, &doc);
    }

    let total_elapsed = started_at.elapsed().as_secs_f64();
    if timed_out {
        println!(
            "stopped total_iterations={} final_infosets={} elapsed={} reason=timeout",
            trainer.iterations,
            trainer.infoset_count(),
            format_duration(total_elapsed)
        );
    } else {
        println!(
            "completed total_iterations={} final_infosets={} elapsed={}",
            trainer.iterations,
            trainer.infoset_count(),
            format_duration(total_elapsed)
        );
    }
    flush_stdout();
}

fn main() {
    let builder = std::thread::Builder::new()
        .name("train-main".to_string())
        .stack_size(64 * 1024 * 1024);
    let handle = builder
        .spawn(train_main)
        .unwrap_or_else(|err| panic!("failed to spawn training thread: {err}"));
    handle
        .join()
        .unwrap_or_else(|_| panic!("training thread panicked"));
}
