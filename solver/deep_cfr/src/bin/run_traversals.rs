use cfr::nlhe_game::NlheGameModel;
use deep_cfr::batched_policy::{BatchInferenceServer, BatchRuntimeConfig};
use deep_cfr::onnx_policy::{OnnxPolicy, PolicyOutput};
use deep_cfr::sample::{write_samples, write_strategy_samples, AdvantageSample, StrategySample};
use deep_cfr::traverse::{
    run_traversal_batch, take_opponent_preflop_debug_snapshot,
    take_traverser_preflop_debug_snapshot, TraversalStats,
};
use game::NlheConfig;
use rayon::prelude::*;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_TRAVERSALS: usize = 5_000;
const DEFAULT_PROGRESS_BATCH: usize = 500;
const DEFAULT_WORKERS: usize = 0;
const DEFAULT_DECK_SAMPLES: usize = 200;
const DEFAULT_NUM_PLAYERS: usize = 2;
const DEFAULT_STARTING_STACK: u32 = 2_000;
const DEFAULT_SMALL_BLIND: u32 = 10;
const DEFAULT_BIG_BLIND: u32 = 20;

#[derive(Debug)]
struct Config {
    onnx: Option<PathBuf>,
    onnx_p0: Option<PathBuf>,
    onnx_p1: Option<PathBuf>,
    onnx_seat_models: Option<Vec<PathBuf>>,
    advantage_samples_out: PathBuf,
    strategy_samples_out: PathBuf,
    player_list: Vec<usize>,
    advantage_samples_template: Option<String>,
    strategy_samples_template: Option<String>,
    traversals: usize,
    progress_batch: usize,
    workers: usize,
    gpu_batch: bool,
    gpu_batch_size: usize,
    gpu_batch_timeout_us: u64,
    gpu_batch_queue_capacity: usize,
    gpu_batch_cuda: bool,
    gpu_batch_tf32: bool,
    gpu_device_id: i32,
    cpu_affinity: Option<Vec<usize>>,
    iteration: u32,
    seed: u64,
    cluster_dir: PathBuf,
    deck_samples: usize,
    num_players: usize,
    starting_stack: u32,
    small_blind: u32,
    big_blind: u32,
}

type AppResult<T> = Result<T, String>;

fn flag_value<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.windows(2).find_map(|window| {
        if window[0] == name {
            Some(window[1].as_str())
        } else {
            None
        }
    })
}

fn parse_optional_path_arg(args: &[String], name: &str) -> Option<PathBuf> {
    flag_value(args, name).map(PathBuf::from)
}

fn parse_optional_path_list_arg(args: &[String], name: &str) -> Option<Vec<PathBuf>> {
    let raw = flag_value(args, name)?;
    let mut out = Vec::new();
    for chunk in raw.split(',') {
        let trimmed = chunk.trim();
        if !trimmed.is_empty() {
            out.push(PathBuf::from(trimmed));
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
}

fn parse_optional_string_arg(args: &[String], name: &str) -> Option<String> {
    flag_value(args, name).map(ToOwned::to_owned)
}

fn parse_optional_usize_list_arg(args: &[String], name: &str) -> AppResult<Option<Vec<usize>>> {
    let raw = match flag_value(args, name) {
        Some(value) => value,
        None => return Ok(None),
    };
    let mut out = Vec::new();
    for chunk in raw.split(',') {
        let trimmed = chunk.trim();
        if trimmed.is_empty() {
            continue;
        }
        out.push(
            trimmed
                .parse::<usize>()
                .map_err(|err| format!("invalid value in {name}: {trimmed} ({err})"))?,
        );
    }
    if out.is_empty() {
        Ok(None)
    } else {
        Ok(Some(out))
    }
}

fn parse_cpu_affinity_arg(args: &[String]) -> AppResult<Option<Vec<usize>>> {
    let raw = match flag_value(args, "--cpu-affinity") {
        Some(value) => value.trim(),
        None => return Ok(None),
    };
    if raw.is_empty() {
        return Ok(None);
    }
    let mut out = Vec::new();
    for chunk in raw.split(',') {
        let token = chunk.trim();
        if token.is_empty() {
            continue;
        }
        if let Some((start_raw, end_raw)) = token.split_once('-') {
            let start = start_raw
                .trim()
                .parse::<usize>()
                .map_err(|err| format!("invalid cpu affinity range start '{start_raw}': {err}"))?;
            let end = end_raw
                .trim()
                .parse::<usize>()
                .map_err(|err| format!("invalid cpu affinity range end '{end_raw}': {err}"))?;
            if end < start {
                return Err(format!(
                    "invalid cpu affinity range '{token}': end must be >= start"
                ));
            }
            for core in start..=end {
                out.push(core);
            }
        } else {
            out.push(
                token
                    .parse::<usize>()
                    .map_err(|err| format!("invalid cpu affinity core '{token}': {err}"))?,
            );
        }
    }
    out.sort_unstable();
    out.dedup();
    if out.is_empty() {
        Ok(None)
    } else {
        Ok(Some(out))
    }
}

fn parse_path_arg(args: &[String], name: &str, default: &str) -> PathBuf {
    PathBuf::from(flag_value(args, name).unwrap_or(default))
}

fn parse_usize_arg(args: &[String], name: &str, default: usize) -> AppResult<usize> {
    match flag_value(args, name) {
        Some(raw) => raw
            .parse::<usize>()
            .map_err(|err| format!("invalid value for {name}: {raw} ({err})")),
        None => Ok(default),
    }
}

fn parse_u32_arg(args: &[String], name: &str, default: u32) -> AppResult<u32> {
    match flag_value(args, name) {
        Some(raw) => raw
            .parse::<u32>()
            .map_err(|err| format!("invalid value for {name}: {raw} ({err})")),
        None => Ok(default),
    }
}

fn parse_u64_arg(args: &[String], name: &str, default: u64) -> AppResult<u64> {
    match flag_value(args, name) {
        Some(raw) => raw
            .parse::<u64>()
            .map_err(|err| format!("invalid value for {name}: {raw} ({err})")),
        None => Ok(default),
    }
}

fn parse_config(args: &[String]) -> AppResult<Config> {
    let onnx = parse_optional_path_arg(args, "--onnx");
    let onnx_p0 = parse_optional_path_arg(args, "--onnx-p0");
    let onnx_p1 = parse_optional_path_arg(args, "--onnx-p1");
    let onnx_seat_models = parse_optional_path_list_arg(args, "--onnx-seat-models");
    let advantage_samples_out = flag_value(args, "--adv-samples-out")
        .or_else(|| flag_value(args, "--samples-out"))
        .map(PathBuf::from)
        .ok_or_else(|| {
            "missing required flag: --adv-samples-out (or legacy --samples-out)".to_string()
        })?;
    let strategy_samples_out = parse_optional_path_arg(args, "--strategy-samples-out")
        .unwrap_or_else(|| default_strategy_samples_out(&advantage_samples_out));
    let player = parse_usize_arg(args, "--player", 0)?;
    let player_list_arg = parse_optional_usize_list_arg(args, "--player-list")?;
    let advantage_samples_template = parse_optional_string_arg(args, "--adv-samples-template");
    let strategy_samples_template = parse_optional_string_arg(args, "--strategy-samples-template");
    let traversals = parse_usize_arg(args, "--traversals", DEFAULT_TRAVERSALS)?;
    let progress_batch = parse_usize_arg(args, "--progress-batch", DEFAULT_PROGRESS_BATCH)?;
    let workers = parse_usize_arg(args, "--workers", DEFAULT_WORKERS)?;
    let gpu_batch = has_flag(args, "--gpu-batch");
    let gpu_batch_size = parse_usize_arg(args, "--gpu-batch-size", 256)?;
    let gpu_batch_timeout_us = parse_u64_arg(args, "--gpu-batch-timeout-us", 500)?;
    let gpu_batch_queue_capacity = parse_usize_arg(args, "--gpu-batch-queue-capacity", 8192)?;
    let gpu_batch_cuda = !has_flag(args, "--gpu-batch-cpu-only");
    let gpu_batch_tf32 = !has_flag(args, "--gpu-batch-no-tf32");
    let gpu_device_id = parse_usize_arg(args, "--gpu-device-id", 0)? as i32;
    let cpu_affinity = parse_cpu_affinity_arg(args)?;
    let iteration = parse_u32_arg(args, "--iteration", 1)?;
    let seed = parse_u64_arg(args, "--seed", 21)?;
    let cluster_dir = parse_path_arg(args, "--cluster-dir", "checkpoints/nlhe_clusters");
    let deck_samples = parse_usize_arg(args, "--deck-samples", DEFAULT_DECK_SAMPLES)?;
    let num_players = parse_usize_arg(args, "--num-players", DEFAULT_NUM_PLAYERS)?;
    let starting_stack = parse_u32_arg(args, "--starting-stack", DEFAULT_STARTING_STACK)?;
    let small_blind = parse_u32_arg(args, "--small-blind", DEFAULT_SMALL_BLIND)?;
    let big_blind = parse_u32_arg(args, "--big-blind", DEFAULT_BIG_BLIND)?;

    if traversals == 0 {
        return Err("--traversals must be greater than 0".to_string());
    }
    if progress_batch == 0 {
        return Err("--progress-batch must be greater than 0".to_string());
    }
    if deck_samples == 0 {
        return Err("--deck-samples must be greater than 0".to_string());
    }
    if num_players < 2 {
        return Err("--num-players must be at least 2".to_string());
    }
    if player >= num_players {
        return Err(format!(
            "--player index {player} is out of range for --num-players {num_players}"
        ));
    }
    let player_list = player_list_arg.unwrap_or_else(|| vec![player]);
    if player_list.is_empty() {
        return Err("--player-list must contain at least one player index".to_string());
    }
    for entry in &player_list {
        if *entry >= num_players {
            return Err(format!(
                "--player-list entry {entry} is out of range for --num-players {num_players}"
            ));
        }
    }
    if player_list.len() > 1 {
        if advantage_samples_template.is_none() {
            return Err(
                "--adv-samples-template is required when --player-list contains multiple players"
                    .to_string(),
            );
        }
        if strategy_samples_template.is_none() {
            return Err(
                "--strategy-samples-template is required when --player-list contains multiple players"
                    .to_string(),
            );
        }
        if !advantage_samples_template
            .as_ref()
            .map(|value| value.contains("{player}"))
            .unwrap_or(false)
        {
            return Err(
                "--adv-samples-template must include '{player}' when --player-list contains multiple players"
                    .to_string(),
            );
        }
        if !strategy_samples_template
            .as_ref()
            .map(|value| value.contains("{player}"))
            .unwrap_or(false)
        {
            return Err(
                "--strategy-samples-template must include '{player}' when --player-list contains multiple players"
                    .to_string(),
            );
        }
    }
    if gpu_batch_size == 0 {
        return Err("--gpu-batch-size must be greater than 0".to_string());
    }
    if gpu_batch_queue_capacity == 0 {
        return Err("--gpu-batch-queue-capacity must be greater than 0".to_string());
    }
    if let Some(paths) = onnx_seat_models.as_ref() {
        if paths.len() != num_players {
            return Err(format!(
                "--onnx-seat-models must contain exactly --num-players ({num_players}) paths, got {}",
                paths.len()
            ));
        }
    } else if onnx.is_none() {
        let has_split_hu = onnx_p0.is_some() && onnx_p1.is_some() && num_players == 2;
        if !has_split_hu {
            return Err(
                "missing policy model path(s): provide --onnx (shared), --onnx-seat-models (comma list), or HU legacy --onnx-p0/--onnx-p1".to_string(),
            );
        }
    }
    if big_blind == 0 {
        return Err("--big-blind must be greater than 0".to_string());
    }
    if small_blind == 0 {
        return Err("--small-blind must be greater than 0".to_string());
    }
    if starting_stack == 0 {
        return Err("--starting-stack must be greater than 0".to_string());
    }

    Ok(Config {
        onnx,
        onnx_p0,
        onnx_p1,
        onnx_seat_models,
        advantage_samples_out,
        strategy_samples_out,
        player_list,
        advantage_samples_template,
        strategy_samples_template,
        traversals,
        progress_batch,
        workers,
        gpu_batch,
        gpu_batch_size,
        gpu_batch_timeout_us,
        gpu_batch_queue_capacity,
        gpu_batch_cuda,
        gpu_batch_tf32,
        gpu_device_id,
        cpu_affinity,
        iteration,
        seed,
        cluster_dir,
        deck_samples,
        num_players,
        starting_stack,
        small_blind,
        big_blind,
    })
}

fn default_strategy_samples_out(advantage_path: &std::path::Path) -> PathBuf {
    let parent = advantage_path
        .parent()
        .map(PathBuf::from)
        .unwrap_or_default();
    let stem = advantage_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("samples");
    let ext = advantage_path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("bin");
    parent.join(format!("{stem}.strategy.{ext}"))
}

fn policy_paths_for_players(config: &Config) -> AppResult<Vec<PathBuf>> {
    if let Some(paths) = &config.onnx_seat_models {
        if paths.len() == config.num_players {
            return Ok(paths.clone());
        }
    }
    if let Some(shared) = &config.onnx {
        return Ok(vec![shared.clone(); config.num_players]);
    }
    if let (Some(p0), Some(p1)) = (&config.onnx_p0, &config.onnx_p1) {
        if config.num_players == 2 {
            return Ok(vec![p0.clone(), p1.clone()]);
        }
    }
    Err(format!(
        "unable to resolve policy paths for {} players",
        config.num_players
    ))
}

fn build_deck_seeds(deck_samples: usize, seed: u64) -> Vec<u64> {
    (0..deck_samples)
        .map(|idx| seed ^ ((idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)))
        .collect()
}

fn format_seconds(seconds: f64) -> String {
    if seconds < 0.05 {
        "0.0s".to_string()
    } else {
        format!("{seconds:.1}s")
    }
}

fn resolve_worker_count(requested_workers: usize, traversals: usize) -> usize {
    let available = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let desired = if requested_workers == 0 {
        available
    } else {
        requested_workers
    };
    desired.max(1).min(traversals.max(1))
}

fn resolve_affinity_cores(requested: &Option<Vec<usize>>) -> Option<Vec<core_affinity::CoreId>> {
    let requested = requested.as_ref()?;
    let available = core_affinity::get_core_ids()?;
    let mut out = Vec::new();
    for core_id in requested {
        if let Some(found) = available.iter().copied().find(|core| core.id == *core_id) {
            out.push(found);
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn render_player_path(template: &str, player: usize) -> PathBuf {
    PathBuf::from(template.replace("{player}", &player.to_string()))
}

fn flush_stdout() {
    let _ = std::io::stdout().flush();
}

#[derive(Debug, Clone, Default)]
struct WorkerTraversalRuntime {
    worker_idx: usize,
    jobs_completed: usize,
    traversals_completed: usize,
    policy_load_sec: f64,
    traverse_sec: f64,
    advantage_samples: usize,
    strategy_samples: usize,
}

#[derive(Debug)]
struct ParallelTraversalOutcome {
    advantage_samples: Vec<AdvantageSample>,
    strategy_samples: Vec<StrategySample>,
    stats: TraversalStats,
    worker_runtime: Vec<WorkerTraversalRuntime>,
    job_chunk_size: usize,
}

fn default_parallel_job_chunk_size(traversals: usize, workers: usize) -> usize {
    let workers = workers.max(1);
    let target_jobs = workers.saturating_mul(8).max(workers);
    ((traversals + target_jobs - 1) / target_jobs).max(1)
}

fn maybe_print_parallel_progress(
    done: usize,
    total: usize,
    progress_step: usize,
    next_progress: &AtomicUsize,
    started_at: Instant,
) {
    if total == 0 {
        return;
    }
    loop {
        let target = next_progress.load(Ordering::Relaxed);
        if done < target || target > total {
            break;
        }
        let mut next_target = target
            .saturating_add(progress_step.max(1))
            .max(target.saturating_add(1));
        if next_target > total {
            next_target = total.saturating_add(1);
        }
        if next_progress
            .compare_exchange(target, next_target, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            let elapsed = started_at.elapsed().as_secs_f64();
            println!(
                "[deep-cfr-traverse] progress: {}/{} traversals {}",
                done.min(total),
                total,
                format_seconds(elapsed)
            );
            flush_stdout();
            break;
        }
    }
}

fn append_debug_log(message: &str, data: &str) {
    let Some(path) = std::env::var_os("TALIBUS_DEBUG_LOG") else {
        return;
    };
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let line = format!(
        "{{\"component\":\"run_traversals\",\"message\":\"{}\",\"data\":{},\"timestamp\":{}}}\n",
        message, data, timestamp
    );
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut file| file.write_all(line.as_bytes()));
}

fn print_help() {
    println!(
        "run_traversals options:\n\
         --onnx <path> (shared policy model for all seats)\n\
         --onnx-seat-models <p0,p1,...> (optional per-seat policy paths; must match --num-players)\n\
         --onnx-p0 <path> (legacy HU path for player 0)\n\
         --onnx-p1 <path> (legacy HU path for player 1)\n\
         --adv-samples-out <path> (required; legacy alias: --samples-out)\n\
         --strategy-samples-out <path> (optional; defaults beside advantage file)\n\
         --adv-samples-template <path-template> (required with --player-list multi-seat; use {{player}})\n\
         --strategy-samples-template <path-template> (required with --player-list multi-seat; use {{player}})\n\
         --player <usize> (default 0)\n\
         --player-list <p0,p1,...> (optional; multi-player consolidation)\n\
         --traversals <usize> (default {DEFAULT_TRAVERSALS})\n\
         --progress-batch <usize> (default {DEFAULT_PROGRESS_BATCH})\n\
         --workers <usize> (default {DEFAULT_WORKERS}, 0=auto)\n\
         --gpu-batch (optional; enables batched inference coordinator)\n\
         --gpu-batch-size <usize> (default 256)\n\
         --gpu-batch-timeout-us <u64> (default 500)\n\
         --gpu-batch-queue-capacity <usize> (default 8192)\n\
         --gpu-batch-cpu-only (optional; batched mode with CPU EP)\n\
         --gpu-batch-no-tf32 (optional; disable CUDA TF32)\n\
         --gpu-device-id <usize> (default 0)\n\
         --cpu-affinity <csv/ranges> (optional; e.g. 0-15)\n\
         --iteration <u32> (default 1)\n\
         --seed <u64> (default 21)\n\
         --cluster-dir <path> (default checkpoints/nlhe_clusters)\n\
         --deck-samples <usize> (default {DEFAULT_DECK_SAMPLES})\n\
         --num-players <usize> (default {DEFAULT_NUM_PLAYERS})\n\
         --starting-stack <u32> (default {DEFAULT_STARTING_STACK})\n\
         --small-blind <u32> (default {DEFAULT_SMALL_BLIND})\n\
         --big-blind <u32> (default {DEFAULT_BIG_BLIND})"
    );
}

fn run_parallel_traversal_batch<P, F>(
    model: &NlheGameModel,
    player: usize,
    traversals: usize,
    seed: u64,
    iteration: u32,
    worker_count: usize,
    progress_batch: usize,
    pool: &rayon::ThreadPool,
    build_worker_policies: &F,
) -> AppResult<ParallelTraversalOutcome>
where
    P: deep_cfr::traverse::PolicyProvider + Send,
    F: Fn(usize) -> AppResult<Vec<P>> + Sync,
{
    let workers = worker_count.min(traversals.max(1)).max(1);
    let job_chunk_size = default_parallel_job_chunk_size(traversals, workers);
    let progress_step = progress_batch.max(1).min(traversals.max(1));
    let started_at = Instant::now();
    let next_start = AtomicUsize::new(0);
    let completed_traversals = AtomicUsize::new(0);
    let next_progress = AtomicUsize::new(progress_step);

    let mut chunks = pool.install(|| {
        (0..workers)
            .into_par_iter()
            .map(
                |worker_idx| -> AppResult<(
                    Vec<AdvantageSample>,
                    Vec<StrategySample>,
                    TraversalStats,
                    WorkerTraversalRuntime,
                )> {
                    let load_started = Instant::now();
                    let mut worker_policies = build_worker_policies(worker_idx)?;
                    let policy_load_sec = load_started.elapsed().as_secs_f64();
                    let traverse_started = Instant::now();
                    let mut all_adv_samples = Vec::new();
                    let mut all_strategy_samples = Vec::new();
                    let mut all_stats = TraversalStats::default();
                    let mut jobs_completed = 0usize;
                    let mut traversals_completed = 0usize;
                    loop {
                        let traversals_done_before =
                            next_start.fetch_add(job_chunk_size, Ordering::Relaxed);
                        if traversals_done_before >= traversals {
                            break;
                        }
                        let worker_traversals =
                            (traversals - traversals_done_before).min(job_chunk_size);
                        let worker_seed = seed
                            ^ ((traversals_done_before as u64 + 1)
                                .wrapping_mul(0x9E37_79B9_7F4A_7C15))
                            ^ ((worker_traversals as u64).wrapping_mul(0xD1B5_4A32_D192_ED03))
                            ^ ((iteration as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
                        let (mut advantage_samples, mut strategy_samples, stats) =
                            run_traversal_batch(
                                model,
                                worker_policies.as_mut_slice(),
                                player,
                                worker_traversals,
                                worker_seed,
                                iteration,
                            )
                            .map_err(|err| {
                                format!("worker {worker_idx} traversal failed: {err}")
                            })?;
                        all_adv_samples.append(&mut advantage_samples);
                        all_strategy_samples.append(&mut strategy_samples);
                        all_stats.merge_from(&stats);
                        jobs_completed += 1;
                        traversals_completed += worker_traversals;
                        let done = completed_traversals
                            .fetch_add(worker_traversals, Ordering::Relaxed)
                            + worker_traversals;
                        maybe_print_parallel_progress(
                            done.min(traversals),
                            traversals,
                            progress_step,
                            &next_progress,
                            started_at,
                        );
                    }
                    let runtime = WorkerTraversalRuntime {
                        worker_idx,
                        jobs_completed,
                        traversals_completed,
                        policy_load_sec,
                        traverse_sec: traverse_started.elapsed().as_secs_f64(),
                        advantage_samples: all_adv_samples.len(),
                        strategy_samples: all_strategy_samples.len(),
                    };
                    Ok((all_adv_samples, all_strategy_samples, all_stats, runtime))
                },
            )
            .collect::<Vec<_>>()
    });
    let mut all_adv_samples = Vec::new();
    let mut all_strategy_samples = Vec::new();
    let mut all_stats = TraversalStats::default();
    let mut worker_runtime = Vec::new();
    for chunk in chunks.drain(..) {
        let (mut advantage_samples, mut strategy_samples, stats, runtime) = chunk?;
        all_adv_samples.append(&mut advantage_samples);
        all_strategy_samples.append(&mut strategy_samples);
        all_stats.merge_from(&stats);
        worker_runtime.push(runtime);
    }
    worker_runtime.sort_by_key(|item| item.worker_idx);
    Ok(ParallelTraversalOutcome {
        advantage_samples: all_adv_samples,
        strategy_samples: all_strategy_samples,
        stats: all_stats,
        worker_runtime,
        job_chunk_size,
    })
}

fn derive_player_seed(base_seed: u64, player: usize, iteration: u32) -> u64 {
    base_seed
        ^ ((player as u64 + 1).wrapping_mul(0xA24B_AED4_0B7F_4D95))
        ^ ((iteration as u64).wrapping_mul(0x94D0_49BB_1331_11EB))
}

fn ensure_output_parent(path: &PathBuf, label: &str) -> AppResult<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed to create {label} output directory {}: {err}",
                    parent.display()
                )
            })?;
        }
    }
    Ok(())
}

fn run_for_player(
    config: &Config,
    model: &NlheGameModel,
    policy_paths: &[PathBuf],
    worker_count: usize,
    parallel_pool: Option<&rayon::ThreadPool>,
    batch_servers: Option<&[BatchInferenceServer]>,
    player: usize,
    advantage_samples_out: &PathBuf,
    strategy_samples_out: &PathBuf,
) -> AppResult<()> {
    println!(
        "[deep-cfr-traverse] player={} iteration={} traversals={} adv_out={} strategy_out={}",
        player,
        config.iteration,
        config.traversals,
        advantage_samples_out.display(),
        strategy_samples_out.display()
    );
    flush_stdout();
    let player_seed = derive_player_seed(config.seed, player, config.iteration);
    let started_at = Instant::now();
    let mut all_adv_samples = Vec::new();
    let mut all_strategy_samples = Vec::new();
    let mut all_stats = TraversalStats::default();

    if let Some(pool) = parallel_pool {
        let parallel = if let Some(servers) = batch_servers {
            run_parallel_traversal_batch(
                model,
                player,
                config.traversals,
                player_seed,
                config.iteration,
                worker_count,
                config.progress_batch,
                pool,
                &|_worker_idx| {
                    Ok(servers
                        .iter()
                        .map(|server| server.client(PolicyOutput::Advantage))
                        .collect::<Vec<_>>())
                },
            )
        } else {
            run_parallel_traversal_batch(
                model,
                player,
                config.traversals,
                player_seed,
                config.iteration,
                worker_count,
                config.progress_batch,
                pool,
                &|worker_idx| {
                    let mut worker_policies = Vec::with_capacity(policy_paths.len());
                    for (policy_idx, policy_path) in policy_paths.iter().enumerate() {
                        let policy = OnnxPolicy::from_file(policy_path).map_err(|err| {
                            format!(
                                "worker {worker_idx} failed to load ONNX policy p{policy_idx} {}: {err}",
                                policy_path.display()
                            )
                        })?;
                        worker_policies.push(policy);
                    }
                    Ok(worker_policies)
                },
            )
        }
        .map_err(|err| format!("parallel traversal failed for player {player}: {err}"))?;
        all_adv_samples = parallel.advantage_samples;
        all_strategy_samples = parallel.strategy_samples;
        all_stats.merge_from(&parallel.stats);
        let mut load_total = 0.0f64;
        let mut traverse_total = 0.0f64;
        let mut min_traversals = usize::MAX;
        let mut max_traversals = 0usize;
        for runtime in &parallel.worker_runtime {
            load_total += runtime.policy_load_sec;
            traverse_total += runtime.traverse_sec;
            min_traversals = min_traversals.min(runtime.traversals_completed);
            max_traversals = max_traversals.max(runtime.traversals_completed);
            println!(
                "[deep-cfr-traverse] player={} worker={} jobs={} traversals={} load={:.3}s run={:.3}s adv_samples={} strategy_samples={}",
                player,
                runtime.worker_idx,
                runtime.jobs_completed,
                runtime.traversals_completed,
                runtime.policy_load_sec,
                runtime.traverse_sec,
                runtime.advantage_samples,
                runtime.strategy_samples
            );
        }
        let min_traversals = if min_traversals == usize::MAX {
            0
        } else {
            min_traversals
        };
        let imbalance = if min_traversals > 0 {
            max_traversals as f64 / min_traversals as f64
        } else {
            0.0
        };
        println!(
            "[deep-cfr-traverse] player={} parallel summary: job_chunk={} worker_load_total={:.3}s worker_run_total={:.3}s traversals_min={} traversals_max={} imbalance={:.3}",
            player,
            parallel.job_chunk_size,
            load_total,
            traverse_total,
            min_traversals,
            max_traversals,
            imbalance
        );
        flush_stdout();
    } else {
        if let Some(servers) = batch_servers {
            let mut policies = servers
                .iter()
                .map(|server| server.client(PolicyOutput::Advantage))
                .collect::<Vec<_>>();
            let mut traversals_done = 0usize;
            let chunk = config.progress_batch.min(config.traversals).max(1);
            while traversals_done < config.traversals {
                let this_batch = (config.traversals - traversals_done).min(chunk);
                let batch_seed = player_seed
                    ^ ((traversals_done as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03))
                    ^ ((player as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
                let (mut batch_adv_samples, mut batch_strategy_samples, stats) = run_traversal_batch(
                    model,
                    policies.as_mut_slice(),
                    player,
                    this_batch,
                    batch_seed,
                    config.iteration,
                )
                .map_err(|err| {
                    format!(
                        "traversal failed for player {player} after {traversals_done} traversals: {err}"
                    )
                })?;
                all_stats.merge_from(&stats);
                all_adv_samples.append(&mut batch_adv_samples);
                all_strategy_samples.append(&mut batch_strategy_samples);
                traversals_done += this_batch;
                let elapsed = started_at.elapsed().as_secs_f64();
                println!(
                    "[deep-cfr-traverse] player={} progress: {}/{} traversals ({} samples) {}",
                    player,
                    traversals_done,
                    config.traversals,
                    all_adv_samples.len(),
                    format_seconds(elapsed)
                );
                flush_stdout();
            }
        } else {
            let mut policies = Vec::with_capacity(policy_paths.len());
            for (policy_idx, policy_path) in policy_paths.iter().enumerate() {
                let policy = OnnxPolicy::from_file(policy_path).map_err(|err| {
                    format!(
                        "failed to load ONNX policy p{policy_idx} from {}: {err}",
                        policy_path.display()
                    )
                })?;
                policies.push(policy);
            }

            let mut traversals_done = 0usize;
            let chunk = config.progress_batch.min(config.traversals).max(1);
            while traversals_done < config.traversals {
                let this_batch = (config.traversals - traversals_done).min(chunk);
                let batch_seed = player_seed
                    ^ ((traversals_done as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03))
                    ^ ((player as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
                let (mut batch_adv_samples, mut batch_strategy_samples, stats) = run_traversal_batch(
                    model,
                    policies.as_mut_slice(),
                    player,
                    this_batch,
                    batch_seed,
                    config.iteration,
                )
                .map_err(|err| {
                    format!(
                        "traversal failed for player {player} after {traversals_done} traversals: {err}"
                    )
                })?;
                all_stats.merge_from(&stats);
                all_adv_samples.append(&mut batch_adv_samples);
                all_strategy_samples.append(&mut batch_strategy_samples);
                traversals_done += this_batch;
                let elapsed = started_at.elapsed().as_secs_f64();
                println!(
                    "[deep-cfr-traverse] player={} progress: {}/{} traversals ({} samples) {}",
                    player,
                    traversals_done,
                    config.traversals,
                    all_adv_samples.len(),
                    format_seconds(elapsed)
                );
                flush_stdout();
            }
        }
    }

    let elapsed = started_at.elapsed().as_secs_f64();
    println!(
        "[deep-cfr-traverse] player={} done: {} traversals -> {} samples in {}",
        player,
        config.traversals,
        all_adv_samples.len(),
        format_seconds(elapsed)
    );
    append_debug_log(
        "traversal_sample_accounting",
        &format!(
            "{{\"player\":{},\"iteration\":{},\"traversals\":{},\"adv_samples_emitted\":{},\"strategy_samples_emitted\":{},\"traverser_nodes\":{},\"opponent_nodes\":{},\"strategy_samples\":{},\"opponent_actions_sampled\":{},\"chance_nodes\":{},\"terminal_nodes\":{}}}",
            player,
            config.iteration,
            config.traversals,
            all_adv_samples.len(),
            all_strategy_samples.len(),
            all_stats.traverser_nodes,
            all_stats.opponent_nodes,
            all_stats.strategy_samples,
            all_stats.opponent_actions_sampled,
            all_stats.chance_nodes,
            all_stats.terminal_nodes
        ),
    );
    if let Some(snapshot) = take_traverser_preflop_debug_snapshot() {
        append_debug_log(
            "traverser_preflop_best_group_profile",
            &format!(
                "{{\"player\":{},\"iteration\":{},\"total_nodes\":{},\"tier_counts\":{:?},\"best_group_counts\":{:?}}}",
                player,
                config.iteration,
                snapshot.total_nodes,
                snapshot.tier_counts,
                snapshot.best_group_counts
            ),
        );
    }
    if let Some(snapshot) = take_opponent_preflop_debug_snapshot() {
        append_debug_log(
            "opponent_preflop_action_mass_profile",
            &format!(
                "{{\"player\":{},\"iteration\":{},\"total_nodes\":{},\"tier_counts\":{:?},\"action_mass\":{:?}}}",
                player,
                config.iteration,
                snapshot.total_nodes,
                snapshot.tier_counts,
                snapshot.action_mass
            ),
        );
    }
    flush_stdout();

    write_samples(advantage_samples_out, &all_adv_samples).map_err(|err| {
        format!(
            "failed to write {} samples to {}: {err}",
            all_adv_samples.len(),
            advantage_samples_out.display()
        )
    })?;
    write_strategy_samples(strategy_samples_out, &all_strategy_samples).map_err(|err| {
        format!(
            "failed to write {} strategy samples to {}: {err}",
            all_strategy_samples.len(),
            strategy_samples_out.display()
        )
    })?;
    println!(
        "[deep-cfr-traverse] player={} written advantage={} ({} samples) strategy={} ({} samples)",
        player,
        advantage_samples_out.display(),
        all_adv_samples.len(),
        strategy_samples_out.display(),
        all_strategy_samples.len(),
    );
    flush_stdout();
    Ok(())
}

fn run_main() -> AppResult<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_help();
        return Ok(());
    }

    let config = parse_config(&args)?;
    let stack_bb = config.starting_stack as f64 / config.big_blind as f64;
    let policy_paths = policy_paths_for_players(&config)?;
    let policy_desc = policy_paths
        .iter()
        .enumerate()
        .map(|(idx, path)| format!("p{idx}={}", path.display()))
        .collect::<Vec<_>>()
        .join(" ");
    let players_desc = config
        .player_list
        .iter()
        .map(|player| player.to_string())
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "[deep-cfr-traverse] players=[{}] iteration={} traversals={} policies={}",
        players_desc, config.iteration, config.traversals, policy_desc,
    );
    println!(
        "[deep-cfr-traverse] game: {}-player {:.1}bb (sb={} bb={} stack={})",
        config.num_players, stack_bb, config.small_blind, config.big_blind, config.starting_stack
    );
    let worker_count = resolve_worker_count(config.workers, config.traversals);
    println!(
        "[deep-cfr-traverse] workers={} progress_batch={}",
        worker_count, config.progress_batch
    );
    println!(
        "[deep-cfr-traverse] gpu_batch={} gpu_batch_size={} gpu_batch_timeout_us={} gpu_batch_queue_capacity={} gpu_batch_cuda={} gpu_device_id={} gpu_batch_tf32={}",
        config.gpu_batch,
        config.gpu_batch_size,
        config.gpu_batch_timeout_us,
        config.gpu_batch_queue_capacity,
        config.gpu_batch_cuda,
        config.gpu_device_id,
        config.gpu_batch_tf32
    );
    flush_stdout();

    let deck_seeds = build_deck_seeds(config.deck_samples, config.seed);
    let model = NlheGameModel::new(
        NlheConfig {
            num_players: config.num_players,
            starting_stack: config.starting_stack,
            small_blind: config.small_blind,
            big_blind: config.big_blind,
        },
        deck_seeds,
        &config.cluster_dir,
    )
    .map_err(|err| {
        format!(
            "failed to construct NLHE model with cluster_dir={}: {err}",
            config.cluster_dir.display()
        )
    })?;

    let requested_affinity = config.cpu_affinity.clone();
    let affinity_cores = resolve_affinity_cores(&requested_affinity).map(Arc::new);
    if let Some(requested) = requested_affinity.as_ref() {
        if affinity_cores.is_some() {
            println!("[deep-cfr-traverse] cpu_affinity requested={requested:?}");
        } else {
            println!(
                "[deep-cfr-traverse] cpu_affinity requested={requested:?} (no matching cores resolved)"
            );
        }
    }

    let parallel_pool = if worker_count > 1 {
        let affinity_for_threads = affinity_cores.clone();
        let mut builder = rayon::ThreadPoolBuilder::new();
        builder = builder
            .num_threads(worker_count)
            .stack_size(64 * 1024 * 1024);
        if let Some(cores) = affinity_for_threads {
            builder = builder.start_handler(move |thread_idx| {
                if !cores.is_empty() {
                    let core = cores[thread_idx % cores.len()];
                    let _ = core_affinity::set_for_current(core);
                }
            });
        }
        Some(
            builder
                .build()
                .map_err(|err| format!("failed to build traversal thread pool: {err}"))?,
        )
    } else {
        if let Some(cores) = affinity_cores.as_ref() {
            if !cores.is_empty() {
                let _ = core_affinity::set_for_current(cores[0]);
            }
        }
        None
    };
    let batch_servers = if config.gpu_batch {
        let runtime = BatchRuntimeConfig {
            max_batch_size: config.gpu_batch_size,
            max_wait_us: config.gpu_batch_timeout_us,
            queue_capacity: config.gpu_batch_queue_capacity,
            use_cuda: config.gpu_batch_cuda,
            cuda_device_id: config.gpu_device_id,
            cuda_tf32: config.gpu_batch_tf32,
        };
        let mut servers = Vec::with_capacity(policy_paths.len());
        for (policy_idx, policy_path) in policy_paths.iter().enumerate() {
            let server = BatchInferenceServer::spawn(
                policy_path.clone(),
                PolicyOutput::Advantage,
                runtime.clone(),
            )
            .map_err(|err| {
                format!(
                    "failed to start batched inference server p{policy_idx} from {}: {err}",
                    policy_path.display()
                )
            })?;
            servers.push(server);
        }
        Some(servers)
    } else {
        None
    };

    for player in &config.player_list {
        let advantage_out = if config.player_list.len() > 1 {
            render_player_path(
                config
                    .advantage_samples_template
                    .as_ref()
                    .ok_or_else(|| "missing --adv-samples-template".to_string())?,
                *player,
            )
        } else {
            config.advantage_samples_out.clone()
        };
        let strategy_out = if config.player_list.len() > 1 {
            render_player_path(
                config
                    .strategy_samples_template
                    .as_ref()
                    .ok_or_else(|| "missing --strategy-samples-template".to_string())?,
                *player,
            )
        } else {
            config.strategy_samples_out.clone()
        };
        ensure_output_parent(&advantage_out, "advantage samples")?;
        ensure_output_parent(&strategy_out, "strategy samples")?;
        run_for_player(
            &config,
            &model,
            &policy_paths,
            worker_count,
            parallel_pool.as_ref(),
            batch_servers.as_deref(),
            *player,
            &advantage_out,
            &strategy_out,
        )?;
    }

    Ok(())
}

fn main() {
    let builder = std::thread::Builder::new()
        .name("deep-cfr-traverse-main".to_string())
        .stack_size(64 * 1024 * 1024);
    let handle = builder
        .spawn(run_main)
        .unwrap_or_else(|err| panic!("failed to spawn traversal thread: {err}"));

    match handle.join() {
        Ok(Ok(())) => {}
        Ok(Err(err)) => {
            eprintln!("[deep-cfr-traverse] error: {err}");
            std::process::exit(1);
        }
        Err(_) => panic!("deep-cfr traversal thread panicked"),
    }
}
