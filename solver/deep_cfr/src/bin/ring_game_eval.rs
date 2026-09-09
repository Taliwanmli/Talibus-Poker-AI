use cfr::external_sampling::{GameModel, NodeKind};
use cfr::nlhe_game::{enumerate_action_space, NlheGameModel, NlheState};
use deep_cfr::calling_station_policy::CallingStationPolicy;
use deep_cfr::encoding::encode_nlhe_state;
use deep_cfr::lag_policy::LagPolicy;
use deep_cfr::nit_policy::NitPolicy;
use deep_cfr::onnx_policy::{OnnxPolicy, PolicyOutput};
use deep_cfr::sample::MAX_ACTIONS;
use deep_cfr::tag_policy::TagPolicy;
use game::NlheConfig;
use rand::prelude::{Rng, SeedableRng, StdRng};
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::Instant;

const DEFAULT_CLUSTER_DIR: &str = "checkpoints/nlhe_clusters";
const DEFAULT_HANDS: usize = 1_000;
const DEFAULT_DECK_SAMPLES: usize = 200;
const DEFAULT_WORKERS: usize = 0;
const DEFAULT_PROGRESS_EVERY: usize = 100;
const DEFAULT_STARTING_STACK: u32 = 2_000;
const DEFAULT_SMALL_BLIND: u32 = 10;
const DEFAULT_BIG_BLIND: u32 = 20;
const DEFAULT_NUM_PLAYERS: usize = 6;

type AppResult<T> = Result<T, String>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EvalMode {
    VsOpponent,
    SelfPlay,
}

impl EvalMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::VsOpponent => "vs_opponent",
            Self::SelfPlay => "self_play",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OpponentType {
    Tag,
    Random,
    CallingStation,
    Lag,
    Nit,
}

impl OpponentType {
    fn as_str(self) -> &'static str {
        match self {
            Self::Tag => "tag",
            Self::Random => "random",
            Self::CallingStation => "calling_station",
            Self::Lag => "lag",
            Self::Nit => "nit",
        }
    }
}

#[derive(Debug)]
struct Config {
    mode: EvalMode,
    opponent: OpponentType,
    model_path: PathBuf,
    policy_output: PolicyOutput,
    model_seat: usize,
    num_players: usize,
    cluster_dir: PathBuf,
    hands: usize,
    deck_samples: usize,
    workers: usize,
    progress_every: usize,
    seed: u64,
    starting_stack: u32,
    small_blind: u32,
    big_blind: u32,
}

#[derive(Debug)]
struct RingEvalReport {
    hands: usize,
    model_total_bb: f64,
    per_seat_total_bb: Vec<f64>,
    model_wins: u64,
    model_losses: u64,
    ties: u64,
}

enum PlayerPolicy {
    Onnx(OnnxPolicy),
    Tag(TagPolicy),
    Random(RandomPolicy),
    CallingStation(CallingStationPolicy),
    Lag(LagPolicy),
    Nit(NitPolicy),
}

struct RandomPolicy;

impl RandomPolicy {
    fn new() -> Self {
        Self
    }

    fn get_strategy(&mut self, action_count: usize) -> Vec<f64> {
        if action_count == 0 {
            return Vec::new();
        }
        let uniform = 1.0 / action_count as f64;
        vec![uniform; action_count]
    }
}

impl PlayerPolicy {
    fn get_strategy(
        &mut self,
        state: &NlheState,
        player_idx: usize,
        action_space: &[cfr::nlhe_game::IndexedAction],
    ) -> AppResult<Vec<f64>> {
        let strategy = match self {
            Self::Onnx(policy) => {
                let features = encode_nlhe_state(state, player_idx)
                    .ok_or_else(|| "expected resolved NLHE state for encoding".to_string())?;
                let (action_slots, action_mask) = action_slots_and_mask(action_space)?;
                policy
                    .get_strategy(&features, &action_slots, &action_mask)
                    .map_err(|err| {
                        format!("failed to query ONNX strategy for player {player_idx}: {err}")
                    })?
            }
            Self::Tag(policy) => policy.get_strategy(state, player_idx)?,
            Self::Random(policy) => policy.get_strategy(action_space.len()),
            Self::CallingStation(policy) => policy.get_strategy(state, player_idx)?,
            Self::Lag(policy) => policy.get_strategy(state, player_idx)?,
            Self::Nit(policy) => policy.get_strategy(state, player_idx)?,
        };
        Ok(strategy)
    }
}

fn flag_value<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.windows(2).find_map(|window| {
        if window[0] == name {
            Some(window[1].as_str())
        } else {
            None
        }
    })
}

fn parse_path_arg(args: &[String], name: &str) -> AppResult<PathBuf> {
    let raw = flag_value(args, name).ok_or_else(|| format!("missing required flag: {name}"))?;
    Ok(PathBuf::from(raw))
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

fn parse_policy_output_arg(
    args: &[String],
    name: &str,
    default: PolicyOutput,
) -> AppResult<PolicyOutput> {
    match flag_value(args, name) {
        Some(raw) => PolicyOutput::parse(raw).ok_or_else(|| {
            format!("invalid value for {name}: {raw} (expected one of: advantage, strategy)")
        }),
        None => Ok(default),
    }
}

fn parse_mode(args: &[String]) -> AppResult<EvalMode> {
    let Some(raw) = flag_value(args, "--mode") else {
        return Ok(EvalMode::VsOpponent);
    };
    let normalized = raw.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "vs-opponent" | "vs_opponent" | "vs-random" | "vs_random" | "vs-tag" | "vs_tag" => {
            Ok(EvalMode::VsOpponent)
        }
        "self-play" | "self_play" => Ok(EvalMode::SelfPlay),
        _ => Err(format!(
            "invalid value for --mode: {raw} (expected vs-opponent or self-play)"
        )),
    }
}

fn parse_opponent(args: &[String]) -> AppResult<OpponentType> {
    let Some(raw) = flag_value(args, "--opponent") else {
        return Ok(OpponentType::Tag);
    };
    let normalized = raw.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "tag" => Ok(OpponentType::Tag),
        "random" => Ok(OpponentType::Random),
        "calling-station" | "calling_station" | "callingstation" | "station" => {
            Ok(OpponentType::CallingStation)
        }
        "lag" | "loose-aggressive" | "loose_aggressive" => Ok(OpponentType::Lag),
        "nit" => Ok(OpponentType::Nit),
        _ => Err(format!(
            "invalid value for --opponent: {raw} (expected tag, random, calling-station, lag, or nit)"
        )),
    }
}

fn parse_config(args: &[String]) -> AppResult<Config> {
    let mode = parse_mode(args)?;
    let opponent = parse_opponent(args)?;
    let model_path = parse_path_arg(args, "--model")?;
    let policy_output = parse_policy_output_arg(args, "--policy", PolicyOutput::Strategy)?;
    let model_seat = parse_usize_arg(args, "--model-seat", 0)?;
    let num_players = parse_usize_arg(args, "--num-players", DEFAULT_NUM_PLAYERS)?;
    let cluster_dir = match flag_value(args, "--cluster-dir") {
        Some(raw) => PathBuf::from(raw),
        None => PathBuf::from(DEFAULT_CLUSTER_DIR),
    };
    let hands = parse_usize_arg(args, "--hands", DEFAULT_HANDS)?;
    let deck_samples = parse_usize_arg(args, "--deck-samples", DEFAULT_DECK_SAMPLES)?;
    let workers = parse_usize_arg(args, "--workers", DEFAULT_WORKERS)?;
    let progress_every = parse_usize_arg(args, "--progress-every", DEFAULT_PROGRESS_EVERY)?;
    let seed = parse_u64_arg(args, "--seed", 42)?;
    let starting_stack = parse_u32_arg(args, "--starting-stack", DEFAULT_STARTING_STACK)?;
    let small_blind = parse_u32_arg(args, "--small-blind", DEFAULT_SMALL_BLIND)?;
    let big_blind = parse_u32_arg(args, "--big-blind", DEFAULT_BIG_BLIND)?;

    if hands == 0 {
        return Err("--hands must be greater than 0".to_string());
    }
    if deck_samples == 0 {
        return Err("--deck-samples must be greater than 0".to_string());
    }
    if num_players < 2 || num_players > 6 {
        return Err("--num-players must be in [2, 6]".to_string());
    }
    if model_seat >= num_players {
        return Err(format!(
            "--model-seat must be in [0, {}]",
            num_players.saturating_sub(1)
        ));
    }
    if starting_stack == 0 {
        return Err("--starting-stack must be greater than 0".to_string());
    }
    if small_blind == 0 {
        return Err("--small-blind must be greater than 0".to_string());
    }
    if big_blind == 0 {
        return Err("--big-blind must be greater than 0".to_string());
    }

    Ok(Config {
        mode,
        opponent,
        model_path,
        policy_output,
        model_seat,
        num_players,
        cluster_dir,
        hands,
        deck_samples,
        workers,
        progress_every,
        seed,
        starting_stack,
        small_blind,
        big_blind,
    })
}

fn print_help() {
    println!(
        "ring_game_eval options:\n\
         --model <path> (required)\n\
         --policy <advantage|strategy> (default strategy)\n\
         --mode <vs-opponent|self-play> (default vs-opponent)\n\
         --opponent <tag|random|calling-station|lag|nit> (default tag)\n\
         --model-seat <usize> (default 0)\n\
         --num-players <usize> (default {DEFAULT_NUM_PLAYERS}, supported 2..6)\n\
         --cluster-dir <path> (default {DEFAULT_CLUSTER_DIR})\n\
         --hands <usize> (default {DEFAULT_HANDS})\n\
         --deck-samples <usize> (default {DEFAULT_DECK_SAMPLES})\n\
         --workers <usize> (default {DEFAULT_WORKERS}; 0 means auto)\n\
         --progress-every <usize> (default {DEFAULT_PROGRESS_EVERY}; 0 disables periodic progress)\n\
         --starting-stack <u32> (default {DEFAULT_STARTING_STACK})\n\
         --small-blind <u32> (default {DEFAULT_SMALL_BLIND})\n\
         --big-blind <u32> (default {DEFAULT_BIG_BLIND})\n\
         --seed <u64> (default 42)"
    );
}

fn build_deck_seeds(deck_samples: usize, seed: u64) -> Vec<u64> {
    (0..deck_samples)
        .map(|idx| seed ^ ((idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)))
        .collect()
}

fn resolve_worker_count(requested_workers: usize, hands: usize) -> usize {
    if requested_workers > 0 {
        return requested_workers.max(1).min(hands.max(1));
    }
    std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1)
        .max(1)
        .min(hands.max(1))
}

fn hand_chunk_size(hands: usize, workers: usize) -> usize {
    hands.max(1).div_ceil(workers.max(1))
}

fn format_duration(seconds: f64) -> String {
    let secs = seconds.max(0.0).round() as u64;
    let hours = secs / 3_600;
    let minutes = (secs % 3_600) / 60;
    let rem_secs = secs % 60;
    if hours > 0 {
        format!("{hours}h{minutes:02}m{rem_secs:02}s")
    } else if minutes > 0 {
        format!("{minutes}m{rem_secs:02}s")
    } else {
        format!("{rem_secs}s")
    }
}

fn maybe_log_progress(
    done: usize,
    total: usize,
    started: Instant,
    next_report: &AtomicUsize,
    report_every: usize,
) {
    if report_every == 0 || total == 0 {
        return;
    }
    loop {
        let target = next_report.load(AtomicOrdering::Relaxed);
        if done < total && done < target {
            return;
        }
        let next_target = target.saturating_add(report_every.max(1));
        if next_report
            .compare_exchange(
                target,
                next_target,
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
            )
            .is_ok()
        {
            let done_clamped = done.min(total);
            let pct = (done_clamped as f64) * 100.0 / (total as f64);
            let elapsed_secs = started.elapsed().as_secs_f64();
            let eta_secs = if done_clamped > 0 && done_clamped < total {
                elapsed_secs * (total - done_clamped) as f64 / done_clamped as f64
            } else {
                0.0
            };
            let hands_per_sec = if elapsed_secs > 0.0 {
                done_clamped as f64 / elapsed_secs
            } else {
                0.0
            };
            println!(
                "[ring-eval] progress: {done_clamped}/{total} ({pct:.1}%) elapsed={} eta~{} speed={hands_per_sec:.1} hands/s",
                format_duration(elapsed_secs),
                format_duration(eta_secs),
            );
            return;
        }
    }
}

fn sample_from_probs(probabilities: &[f64], rng: &mut StdRng) -> usize {
    if probabilities.is_empty() {
        return 0;
    }
    let draw: f64 = rng.gen();
    let mut cumulative = 0.0;
    for (idx, prob) in probabilities.iter().enumerate() {
        cumulative += *prob;
        if draw <= cumulative {
            return idx;
        }
    }
    probabilities.len() - 1
}

fn action_slots_and_mask(
    action_space: &[cfr::nlhe_game::IndexedAction],
) -> AppResult<(Vec<usize>, [f32; MAX_ACTIONS])> {
    let mut slots = Vec::with_capacity(action_space.len());
    let mut mask = [0.0f32; MAX_ACTIONS];
    for action in action_space {
        let slot = usize::from(action.policy_slot);
        if slot >= MAX_ACTIONS {
            return Err(format!("invalid policy slot {slot} for action space"));
        }
        if mask[slot] > 0.0 {
            return Err(format!("duplicate policy slot {slot} in action space"));
        }
        mask[slot] = 1.0;
        slots.push(slot);
    }
    Ok((slots, mask))
}

fn build_policies_for_worker(config: &Config, worker_idx: usize) -> AppResult<Vec<PlayerPolicy>> {
    let mut policies = Vec::with_capacity(config.num_players);
    for seat in 0..config.num_players {
        let use_model = match config.mode {
            EvalMode::VsOpponent => seat == config.model_seat,
            EvalMode::SelfPlay => true,
        };
        if use_model {
            let policy =
                OnnxPolicy::from_file_with_output_mode(&config.model_path, config.policy_output)
                    .map_err(|err| {
                        format!(
                            "worker {worker_idx} failed to load model {}: {err}",
                            config.model_path.display()
                        )
                    })?;
            policies.push(PlayerPolicy::Onnx(policy));
        } else {
            match config.opponent {
                OpponentType::Tag => policies.push(PlayerPolicy::Tag(TagPolicy::new())),
                OpponentType::Random => policies.push(PlayerPolicy::Random(RandomPolicy::new())),
                OpponentType::CallingStation => {
                    policies.push(PlayerPolicy::CallingStation(CallingStationPolicy::new()))
                }
                OpponentType::Lag => policies.push(PlayerPolicy::Lag(LagPolicy::new())),
                OpponentType::Nit => policies.push(PlayerPolicy::Nit(NitPolicy::new())),
            }
        }
    }
    Ok(policies)
}

fn simulate_hand(
    game: &NlheGameModel,
    policies: &mut [PlayerPolicy],
    seed: u64,
) -> AppResult<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state: NlheState = game.root_state();
    loop {
        match game.node_kind(&state) {
            NodeKind::Terminal => {
                let per_seat = (0..policies.len())
                    .map(|seat| game.terminal_utility(&state, seat))
                    .collect::<Vec<_>>();
                return Ok(per_seat);
            }
            NodeKind::Chance => {
                let probs = game.chance_probabilities(&state);
                let action_idx = sample_from_probs(&probs, &mut rng);
                state = game.next_state(&state, action_idx);
            }
            NodeKind::Player(player_idx) => {
                let game_state = state
                    .game
                    .as_ref()
                    .ok_or_else(|| "expected resolved NLHE state at player node".to_string())?;
                let action_space = enumerate_action_space(game_state, player_idx);
                let action_count = action_space.len();
                if action_count == 0 {
                    let per_seat = (0..policies.len())
                        .map(|seat| game.terminal_utility(&state, seat))
                        .collect::<Vec<_>>();
                    return Ok(per_seat);
                }

                let strategy = policies[player_idx]
                    .get_strategy(&state, player_idx, &action_space)
                    .map_err(|err| {
                        format!("failed to query strategy for player {player_idx}: {err}")
                    })?;
                if strategy.len() != action_count {
                    return Err(format!(
                        "strategy/action count mismatch: strategy={}, actions={action_count}",
                        strategy.len()
                    ));
                }
                let action_idx = sample_from_probs(&strategy, &mut rng);
                state = game.next_state(&state, action_idx);
            }
        }
    }
}

fn evaluate_ring_game(
    config: &Config,
    game: &NlheGameModel,
    pool: &rayon::ThreadPool,
    worker_count: usize,
) -> AppResult<RingEvalReport> {
    let hands: Vec<usize> = (0..config.hands).collect();
    let chunk_size = hand_chunk_size(hands.len(), worker_count);
    let started = Instant::now();
    let done = AtomicUsize::new(0);
    let next_report = AtomicUsize::new(config.progress_every.max(1));

    println!(
        "[ring-eval] start: mode={} opponent={} hands={} players={} model_seat={} workers={} chunk_size={}",
        config.mode.as_str(),
        config.opponent.as_str(),
        config.hands,
        config.num_players,
        config.model_seat,
        worker_count,
        chunk_size
    );

    let partials = pool.install(|| {
        hands
            .par_chunks(chunk_size)
            .enumerate()
            .map(
                |(worker_idx, chunk)| -> AppResult<(f64, Vec<f64>, u64, u64, u64)> {
                    let mut policies = build_policies_for_worker(config, worker_idx)?;
                    let mut model_total_bb = 0.0f64;
                    let mut per_seat_total_bb = vec![0.0f64; config.num_players];
                    let mut model_wins = 0u64;
                    let mut model_losses = 0u64;
                    let mut ties = 0u64;

                    for hand_idx in chunk {
                        let hand_seed = config.seed
                            ^ ((*hand_idx as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03));
                        let utilities = simulate_hand(game, &mut policies, hand_seed)?;
                        if utilities.len() != config.num_players {
                            return Err(format!(
                                "terminal utility size mismatch: expected {}, got {}",
                                config.num_players,
                                utilities.len()
                            ));
                        }
                        for (seat, value) in utilities.iter().enumerate() {
                            per_seat_total_bb[seat] += *value;
                        }
                        let model_value = utilities[config.model_seat];
                        model_total_bb += model_value;
                        if model_value > 1e-12 {
                            model_wins += 1;
                        } else if model_value < -1e-12 {
                            model_losses += 1;
                        } else {
                            ties += 1;
                        }

                        let done_count = done.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                        maybe_log_progress(
                            done_count,
                            config.hands,
                            started,
                            &next_report,
                            config.progress_every,
                        );
                    }

                    Ok((
                        model_total_bb,
                        per_seat_total_bb,
                        model_wins,
                        model_losses,
                        ties,
                    ))
                },
            )
            .collect::<Vec<_>>()
    });

    let mut model_total_bb = 0.0f64;
    let mut per_seat_total_bb = vec![0.0f64; config.num_players];
    let mut model_wins = 0u64;
    let mut model_losses = 0u64;
    let mut ties = 0u64;

    for partial in partials {
        let (model_total, seat_totals, wins, losses, ties_count) = partial?;
        model_total_bb += model_total;
        for (seat, value) in seat_totals.iter().enumerate() {
            per_seat_total_bb[seat] += *value;
        }
        model_wins += wins;
        model_losses += losses;
        ties += ties_count;
    }

    println!(
        "[ring-eval] done: {} hands in {}",
        config.hands,
        format_duration(started.elapsed().as_secs_f64())
    );
    Ok(RingEvalReport {
        hands: config.hands,
        model_total_bb,
        per_seat_total_bb,
        model_wins,
        model_losses,
        ties,
    })
}

fn format_float_array(values: &[f64], decimals: usize) -> String {
    let items = values
        .iter()
        .map(|value| format!("{value:.decimals$}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
}

fn run_main() -> AppResult<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_help();
        return Ok(());
    }

    let config = parse_config(&args)?;
    let worker_count = resolve_worker_count(config.workers, config.hands);
    println!(
        "[ring-eval] mode={} opponent={} model={} policy={}",
        config.mode.as_str(),
        config.opponent.as_str(),
        config.model_path.display(),
        config.policy_output.as_str()
    );
    println!(
        "[ring-eval] game: {}-player {:.1}bb (sb={} bb={} stack={})",
        config.num_players,
        config.starting_stack as f64 / config.big_blind as f64,
        config.small_blind,
        config.big_blind,
        config.starting_stack
    );
    println!(
        "[ring-eval] hands={} deck_samples={} workers={} seed={} progress_every={}",
        config.hands, config.deck_samples, worker_count, config.seed, config.progress_every
    );

    let deck_seed = config.seed ^ 0xA24B_AED4_963E_E407;
    let deck_seeds = build_deck_seeds(config.deck_samples, deck_seed);
    let game = NlheGameModel::new(
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
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .build()
        .map_err(|err| format!("failed to build rayon thread pool: {err}"))?;

    let started = Instant::now();
    let report = evaluate_ring_game(&config, &game, &pool, worker_count)?;
    let elapsed = started.elapsed().as_secs_f64();
    let model_bb_per_hand = report.model_total_bb / report.hands as f64;
    let model_bb_per_100 = model_bb_per_hand * 100.0;
    let per_seat_bb_per_hand = report
        .per_seat_total_bb
        .iter()
        .map(|value| *value / report.hands as f64)
        .collect::<Vec<_>>();
    let per_seat_bb_per_100 = per_seat_bb_per_hand
        .iter()
        .map(|value| *value * 100.0)
        .collect::<Vec<_>>();
    let zero_sum_check = per_seat_bb_per_hand.iter().sum::<f64>();

    println!("[ring-eval] results:");
    println!(
        "[ring-eval]   model seat {}: {model_bb_per_hand:+.6} bb/hand ({model_bb_per_100:+.3} bb/100)",
        config.model_seat
    );
    println!(
        "[ring-eval]   outcomes: wins={} losses={} ties={}",
        report.model_wins, report.model_losses, report.ties
    );
    println!("[ring-eval]   zero_sum_check={zero_sum_check:+.9} bb/hand");
    println!("[ring-eval]   hands={} elapsed={elapsed:.1}s", report.hands);

    let payload = format!(
        "{{\"mode\":\"{}\",\"opponent\":\"{}\",\"hands\":{},\"num_players\":{},\"model_seat\":{},\"model_bb_per_hand\":{:.6},\"model_bb_per_100\":{:.3},\"per_seat_bb_per_hand\":{},\"per_seat_bb_per_100\":{},\"elapsed_sec\":{:.3},\"model_wins\":{},\"model_losses\":{},\"ties\":{},\"zero_sum_check_bb_per_hand\":{:.9}}}",
        config.mode.as_str(),
        config.opponent.as_str(),
        report.hands,
        config.num_players,
        config.model_seat,
        model_bb_per_hand,
        model_bb_per_100,
        format_float_array(&per_seat_bb_per_hand, 6),
        format_float_array(&per_seat_bb_per_100, 3),
        elapsed,
        report.model_wins,
        report.model_losses,
        report.ties,
        zero_sum_check,
    );
    println!("RING_EVAL_JSON {payload}");
    Ok(())
}

fn main() {
    let builder = std::thread::Builder::new()
        .name("deep-cfr-ring-eval-main".to_string())
        .stack_size(64 * 1024 * 1024);
    let handle = builder
        .spawn(run_main)
        .unwrap_or_else(|err| panic!("failed to spawn ring_game_eval thread: {err}"));

    match handle.join() {
        Ok(Ok(())) => {}
        Ok(Err(err)) => {
            eprintln!("[ring-eval] error: {err}");
            std::process::exit(1);
        }
        Err(_) => panic!("ring_game_eval thread panicked"),
    }
}
