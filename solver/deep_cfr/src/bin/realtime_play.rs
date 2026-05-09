use cfr::external_sampling::{GameModel, NodeKind};
use cfr::nlhe_game::{enumerate_action_space, IndexedAction, NlheGameModel, NlheState};
use deep_cfr::batched_policy::BatchRuntimeConfig;
use deep_cfr::calling_station_policy::CallingStationPolicy;
use deep_cfr::lag_policy::LagPolicy;
use deep_cfr::nit_policy::NitPolicy;
use deep_cfr::onnx_policy::PolicyOutput;
use deep_cfr::realtime_search::{sample_action_from_result, RealtimeConfig, RealtimeSearcher};
use deep_cfr::tag_policy::TagPolicy;
use game::{BettingRound, Card, Deck, NlheConfig, NlheGame, Rank, Suit};
use rand::prelude::{Rng, SeedableRng, StdRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::Instant;

const DEFAULT_CLUSTER_DIR: &str = "checkpoints/nlhe_clusters";
const DEFAULT_NUM_PLAYERS: usize = 6;
const DEFAULT_DECK_SAMPLES: usize = 200;
const DEFAULT_STARTING_STACK: u32 = 2_000;
const DEFAULT_SMALL_BLIND: u32 = 10;
const DEFAULT_BIG_BLIND: u32 = 20;
const DEFAULT_TIME_BUDGET_MS: u64 = 10_000;
const DEFAULT_THREADS: usize = 24;
const DEFAULT_LEAF_ROLLOUTS: usize = 1;
const DEFAULT_HANDS: usize = 50;
const DEFAULT_PROGRESS_EVERY: usize = 5;
const DEFAULT_BATCH_SIZE: usize = 128;
const DEFAULT_BATCH_WAIT_US: u64 = 50;
const DEFAULT_BATCH_QUEUE_CAPACITY: usize = 8_192;
const ADAPTIVE_PREFLOP_NO_RAISE_MS: u64 = 2_000;
const ADAPTIVE_PREFLOP_FACING_RAISE_MS: u64 = 5_000;
const ADAPTIVE_FLOP_MULTIWAY_MS: u64 = 8_000;
const ADAPTIVE_LATE_STREET_MS: u64 = 10_000;

type AppResult<T> = Result<T, String>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Interactive,
    Benchmark,
    RingEval,
    Live,
}

impl Mode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Interactive => "interactive",
            Self::Benchmark => "benchmark",
            Self::RingEval => "ring-eval",
            Self::Live => "live",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RingEvalOpponent {
    Tag,
    CallingStation,
    Lag,
    Nit,
}

impl RingEvalOpponent {
    fn as_str(self) -> &'static str {
        match self {
            Self::Tag => "tag",
            Self::CallingStation => "calling_station",
            Self::Lag => "lag",
            Self::Nit => "nit",
        }
    }
}

#[derive(Debug)]
struct Config {
    mode: Mode,
    opponent: RingEvalOpponent,
    opponents: Vec<RingEvalOpponent>,
    model_path: PathBuf,
    cluster_dir: PathBuf,
    policy_output: PolicyOutput,
    num_players: usize,
    deck_samples: usize,
    seed: u64,
    starting_stack: u32,
    small_blind: u32,
    big_blind: u32,
    time_budget_ms: u64,
    threads: usize,
    max_iterations: usize,
    leaf_rollouts: usize,
    enable_gpu_batch: bool,
    adaptive_budget: bool,
    batch_size: usize,
    batch_wait_us: u64,
    batch_queue_capacity: usize,
    hands: usize,
    model_seat: usize,
    progress_every: usize,
}

#[derive(Debug, Deserialize)]
struct InteractiveRequest {
    hero_seat: Option<usize>,
    deck_index: Option<usize>,
    deck_seed: Option<u64>,
    action_indices: Option<Vec<usize>>,
    time_budget_ms: Option<u64>,
    threads: Option<usize>,
    max_iterations: Option<usize>,
    leaf_rollouts_per_action: Option<usize>,
}

#[derive(Debug, Serialize)]
struct InteractiveActionOut {
    action_index: usize,
    action_token: String,
    policy_slot: usize,
    probability: f64,
}

#[derive(Debug, Serialize)]
struct InteractiveResponse {
    mode: &'static str,
    hero_seat: usize,
    elapsed_ms: u128,
    iterations: usize,
    chosen_action_index: usize,
    chosen_action_token: String,
    action_probabilities: Vec<InteractiveActionOut>,
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

#[derive(Debug, Default)]
struct RingDecisionStats {
    decisions: u64,
    total_search_ms: u128,
    total_iterations: u64,
    total_infosets: u64,
    total_budget_ms: u64,
}

enum OpponentPolicy {
    Tag(TagPolicy),
    CallingStation(CallingStationPolicy),
    Lag(LagPolicy),
    Nit(NitPolicy),
}

impl OpponentPolicy {
    fn from_kind(kind: RingEvalOpponent) -> Self {
        match kind {
            RingEvalOpponent::Tag => Self::Tag(TagPolicy::new()),
            RingEvalOpponent::CallingStation => Self::CallingStation(CallingStationPolicy::new()),
            RingEvalOpponent::Lag => Self::Lag(LagPolicy::new()),
            RingEvalOpponent::Nit => Self::Nit(NitPolicy::new()),
        }
    }

    fn get_strategy(&mut self, state: &NlheState, player_idx: usize) -> AppResult<Vec<f64>> {
        match self {
            Self::Tag(policy) => policy
                .get_strategy(state, player_idx)
                .map_err(|err| format!("tag policy query failed for seat {player_idx}: {err}")),
            Self::CallingStation(policy) => policy.get_strategy(state, player_idx).map_err(|err| {
                format!("calling-station policy query failed for seat {player_idx}: {err}")
            }),
            Self::Lag(policy) => policy
                .get_strategy(state, player_idx)
                .map_err(|err| format!("lag policy query failed for seat {player_idx}: {err}")),
            Self::Nit(policy) => policy
                .get_strategy(state, player_idx)
                .map_err(|err| format!("nit policy query failed for seat {player_idx}: {err}")),
        }
    }
}

fn format_opponents(opponents: &[RingEvalOpponent]) -> String {
    opponents
        .iter()
        .map(|opponent| opponent.as_str())
        .collect::<Vec<_>>()
        .join(",")
}

struct BenchmarkScenario {
    name: String,
    state: NlheState,
    hero_seat: usize,
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

fn flag_present(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
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

fn parse_mode(args: &[String]) -> AppResult<Mode> {
    let Some(raw) = flag_value(args, "--mode") else {
        return Ok(Mode::Benchmark);
    };
    let normalized = raw.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "interactive" => Ok(Mode::Interactive),
        "benchmark" => Ok(Mode::Benchmark),
        "ring-eval" | "ring_eval" | "ringeval" => Ok(Mode::RingEval),
        "live" => Ok(Mode::Live),
        _ => Err(format!(
            "invalid value for --mode: {raw} (expected interactive|benchmark|ring-eval|live)"
        )),
    }
}

fn parse_ring_eval_opponent(raw: &str) -> AppResult<RingEvalOpponent> {
    let normalized = raw.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "tag" => Ok(RingEvalOpponent::Tag),
        "calling-station" | "calling_station" | "callingstation" | "station" => {
            Ok(RingEvalOpponent::CallingStation)
        }
        "lag" | "loose-aggressive" | "loose_aggressive" => Ok(RingEvalOpponent::Lag),
        "nit" => Ok(RingEvalOpponent::Nit),
        _ => Err(format!(
            "invalid opponent value: {raw} (expected tag|calling-station|lag|nit)"
        )),
    }
}

fn parse_opponent(args: &[String]) -> AppResult<RingEvalOpponent> {
    let Some(raw) = flag_value(args, "--opponent") else {
        return Ok(RingEvalOpponent::Tag);
    };
    parse_ring_eval_opponent(raw).map_err(|_| {
        format!("invalid value for --opponent: {raw} (expected tag|calling-station|lag|nit)")
    })
}

fn parse_opponents_list(
    args: &[String],
    fallback: RingEvalOpponent,
    num_players: usize,
) -> AppResult<Vec<RingEvalOpponent>> {
    let target_len = num_players.saturating_sub(1);
    let Some(raw) = flag_value(args, "--opponents") else {
        return Ok(vec![fallback; target_len]);
    };

    let parsed = raw
        .split(',')
        .filter(|item| !item.trim().is_empty())
        .map(parse_ring_eval_opponent)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("invalid value for --opponents: {err}"))?;

    if parsed.len() != target_len {
        return Err(format!(
            "--opponents must contain exactly {} entries for {} players (excluding model seat), got {}",
            target_len,
            num_players,
            parsed.len()
        ));
    }
    Ok(parsed)
}

fn parse_config(args: &[String]) -> AppResult<Config> {
    let mode = parse_mode(args)?;
    let model_path = parse_path_arg(args, "--model")?;
    let cluster_dir = match flag_value(args, "--cluster-dir") {
        Some(raw) => PathBuf::from(raw),
        None => PathBuf::from(DEFAULT_CLUSTER_DIR),
    };
    let policy_output = parse_policy_output_arg(args, "--policy", PolicyOutput::Strategy)?;
    let num_players = parse_usize_arg(args, "--num-players", DEFAULT_NUM_PLAYERS)?;
    let opponent = parse_opponent(args)?;
    let opponents = parse_opponents_list(args, opponent, num_players)?;
    let deck_samples = parse_usize_arg(args, "--deck-samples", DEFAULT_DECK_SAMPLES)?;
    let seed = parse_u64_arg(args, "--seed", 42)?;
    let starting_stack = parse_u32_arg(args, "--starting-stack", DEFAULT_STARTING_STACK)?;
    let small_blind = parse_u32_arg(args, "--small-blind", DEFAULT_SMALL_BLIND)?;
    let big_blind = parse_u32_arg(args, "--big-blind", DEFAULT_BIG_BLIND)?;
    let time_budget_ms = parse_u64_arg(args, "--time-budget-ms", DEFAULT_TIME_BUDGET_MS)?;
    let threads = parse_usize_arg(args, "--threads", DEFAULT_THREADS)?;
    let max_iterations = parse_usize_arg(args, "--max-iterations", 0)?;
    let leaf_rollouts = parse_usize_arg(args, "--leaf-rollouts", DEFAULT_LEAF_ROLLOUTS)?;
    let batch_size = parse_usize_arg(args, "--batch-size", DEFAULT_BATCH_SIZE)?;
    let batch_wait_us = parse_u64_arg(args, "--batch-wait-us", DEFAULT_BATCH_WAIT_US)?;
    let batch_queue_capacity =
        parse_usize_arg(args, "--batch-queue-capacity", DEFAULT_BATCH_QUEUE_CAPACITY)?;
    let hands = parse_usize_arg(args, "--hands", DEFAULT_HANDS)?;
    let model_seat = parse_usize_arg(args, "--model-seat", 0)?;
    let progress_every = parse_usize_arg(args, "--progress-every", DEFAULT_PROGRESS_EVERY)?;
    let enable_gpu_batch = !flag_present(args, "--disable-gpu-batch");
    let adaptive_budget = !flag_present(args, "--disable-adaptive-budget");

    if num_players < 2 || num_players > 6 {
        return Err("--num-players must be in [2, 6]".to_string());
    }
    if deck_samples == 0 {
        return Err("--deck-samples must be greater than 0".to_string());
    }
    if small_blind == 0 || big_blind == 0 || small_blind >= big_blind {
        return Err("blind structure is invalid; require 0 < small_blind < big_blind".to_string());
    }
    if starting_stack < big_blind {
        return Err("--starting-stack must be >= big_blind".to_string());
    }
    if threads == 0 {
        return Err("--threads must be greater than 0".to_string());
    }
    if batch_size == 0 {
        return Err("--batch-size must be greater than 0".to_string());
    }
    if batch_wait_us == 0 {
        return Err("--batch-wait-us must be greater than 0".to_string());
    }
    if batch_queue_capacity == 0 {
        return Err("--batch-queue-capacity must be greater than 0".to_string());
    }
    if model_seat >= num_players {
        return Err(format!(
            "--model-seat must be in [0, {}]",
            num_players.saturating_sub(1)
        ));
    }
    if hands == 0 {
        return Err("--hands must be greater than 0".to_string());
    }

    Ok(Config {
        mode,
        opponent,
        opponents,
        model_path,
        cluster_dir,
        policy_output,
        num_players,
        deck_samples,
        seed,
        starting_stack,
        small_blind,
        big_blind,
        time_budget_ms,
        threads,
        max_iterations,
        leaf_rollouts,
        enable_gpu_batch,
        adaptive_budget,
        batch_size,
        batch_wait_us,
        batch_queue_capacity,
        hands,
        model_seat,
        progress_every,
    })
}

fn print_help() {
    println!(
        "realtime_play options:\n\
         --model <path> (required)\n\
         --mode <interactive|benchmark|ring-eval|live> (default benchmark)\n\
         --mode live: persistent JSON-line protocol for vision bridge; reads one JSON object per line from stdin, writes one per line to stdout\n\
         --opponent <tag|calling-station|lag|nit> (ring-eval mode, default tag)\n\
         --opponents <csv> (ring-eval mode, optional mixed opponent list; example: tag,calling-station,lag,nit,tag)\n\
         --cluster-dir <path> (default {DEFAULT_CLUSTER_DIR})\n\
         --policy <advantage|strategy> (default strategy)\n\
         --num-players <usize> (default {DEFAULT_NUM_PLAYERS}, supported 2..6)\n\
         --deck-samples <usize> (default {DEFAULT_DECK_SAMPLES})\n\
         --starting-stack <u32> (default {DEFAULT_STARTING_STACK})\n\
         --small-blind <u32> (default {DEFAULT_SMALL_BLIND})\n\
         --big-blind <u32> (default {DEFAULT_BIG_BLIND})\n\
         --seed <u64> (default 42)\n\
         --time-budget-ms <u64> (default {DEFAULT_TIME_BUDGET_MS})\n\
         --threads <usize> (default {DEFAULT_THREADS})\n\
         --max-iterations <usize> (default 0: unlimited under time budget)\n\
         --leaf-rollouts <usize> (default {DEFAULT_LEAF_ROLLOUTS})\n\
         --batch-size <usize> (default {DEFAULT_BATCH_SIZE})\n\
         --batch-wait-us <u64> (default {DEFAULT_BATCH_WAIT_US})\n\
         --batch-queue-capacity <usize> (default {DEFAULT_BATCH_QUEUE_CAPACITY})\n\
         --hands <usize> (ring-eval mode, default {DEFAULT_HANDS})\n\
         --model-seat <usize> (ring-eval mode, default 0)\n\
         --progress-every <usize> (ring-eval mode, default {DEFAULT_PROGRESS_EVERY})\n\
         --disable-gpu-batch (use per-thread ONNX sessions instead of shared GPU batch server)\n\
         --disable-adaptive-budget (disable street-aware budget scheduling)"
    );
}

fn build_deck_seeds(deck_samples: usize, seed: u64) -> Vec<u64> {
    (0..deck_samples)
        .map(|idx| seed ^ ((idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)))
        .collect()
}

fn build_game_model(config: &Config) -> AppResult<NlheGameModel> {
    let deck_seed = config.seed ^ 0xA24B_AED4_963E_E407;
    let deck_seeds = build_deck_seeds(config.deck_samples, deck_seed);
    NlheGameModel::new(
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
    })
}

fn build_search_config(config: &Config) -> RealtimeConfig {
    RealtimeConfig {
        time_budget_ms: config.time_budget_ms,
        worker_threads: config.threads,
        max_iterations: config.max_iterations,
        leaf_rollouts_per_action: config.leaf_rollouts,
        enable_gpu_batch: config.enable_gpu_batch,
        rng_seed: config.seed ^ 0x5AF2_BC42_E908_91E3,
    }
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

fn round_label(state: &NlheState) -> &'static str {
    let Some(game_state) = state.game.as_ref() else {
        return "chance";
    };
    match game_state.round {
        BettingRound::Preflop => "preflop",
        BettingRound::Flop => "flop",
        BettingRound::Turn => "turn",
        BettingRound::River => "river",
        BettingRound::Complete => "complete",
    }
}

fn active_players_for_state(state: &NlheState) -> usize {
    let Some(game_state) = state.game.as_ref() else {
        return 0;
    };
    game_state
        .players
        .iter()
        .filter(|player| !player.folded && !player.all_in)
        .count()
}

fn adaptive_target_budget_ms(state: &NlheState, action_space: &[IndexedAction]) -> u64 {
    let Some(game_state) = state.game.as_ref() else {
        return ADAPTIVE_LATE_STREET_MS;
    };
    let active_players = active_players_for_state(state);
    let facing_raise = action_space.iter().any(|action| action.sort_group == 0);
    match game_state.round {
        BettingRound::Preflop => {
            if facing_raise {
                ADAPTIVE_PREFLOP_FACING_RAISE_MS
            } else {
                ADAPTIVE_PREFLOP_NO_RAISE_MS
            }
        }
        BettingRound::Flop if active_players > 2 => ADAPTIVE_FLOP_MULTIWAY_MS,
        BettingRound::Flop | BettingRound::Turn | BettingRound::River => ADAPTIVE_LATE_STREET_MS,
        BettingRound::Complete => ADAPTIVE_PREFLOP_NO_RAISE_MS,
    }
}

fn resolve_budget_ms(config: &Config, state: &NlheState, action_space: &[IndexedAction]) -> u64 {
    let base_budget_ms = config.time_budget_ms.max(1);
    if !config.adaptive_budget {
        return base_budget_ms;
    }
    adaptive_target_budget_ms(state, action_space).min(base_budget_ms)
}

fn sample_from_probs(probabilities: &[f64], rng: &mut StdRng) -> usize {
    if probabilities.is_empty() {
        return 0;
    }
    let draw = rng.gen::<f64>();
    let mut cumulative = 0.0f64;
    for (idx, prob) in probabilities.iter().enumerate() {
        cumulative += *prob;
        if draw <= cumulative {
            return idx;
        }
    }
    probabilities.len().saturating_sub(1)
}

fn choose_deck_index(request: &InteractiveRequest, model: &NlheGameModel) -> usize {
    if model.deck_seeds.is_empty() {
        return 0;
    }
    if let Some(index) = request.deck_index {
        return index % model.deck_seeds.len();
    }
    if let Some(seed) = request.deck_seed {
        if let Some(index) = model.deck_seeds.iter().position(|value| *value == seed) {
            return index;
        }
        return seed as usize % model.deck_seeds.len();
    }
    0
}

fn build_state_from_interactive_request(
    request: &InteractiveRequest,
    model: &NlheGameModel,
) -> AppResult<(NlheState, usize)> {
    let mut state = model.root_state();
    if matches!(model.node_kind(&state), NodeKind::Chance) {
        let deck_index = choose_deck_index(request, model);
        state = model.next_state(&state, deck_index);
    }

    for action_idx in request.action_indices.clone().unwrap_or_default() {
        match model.node_kind(&state) {
            NodeKind::Terminal => {
                return Err(
                    "action history cannot be applied because the hand is already terminal".to_string()
                )
            }
            NodeKind::Chance => {
                let probs = model.chance_probabilities(&state);
                if probs.is_empty() {
                    return Err("chance node has no probabilities".to_string());
                }
                let idx = action_idx % probs.len();
                state = model.next_state(&state, idx);
            }
            NodeKind::Player(player_idx) => {
                let game_state = state
                    .game
                    .as_ref()
                    .ok_or_else(|| "expected resolved NLHE game while replaying actions".to_string())?;
                let action_space = enumerate_action_space(game_state, player_idx);
                if action_space.is_empty() {
                    return Err("player node has no legal actions".to_string());
                }
                if action_idx >= action_space.len() {
                    return Err(format!(
                        "action index {action_idx} is out of range for player {player_idx} ({} legal actions)",
                        action_space.len()
                    ));
                }
                state = model.next_state(&state, action_idx);
            }
        }
    }

    let actor = match model.node_kind(&state) {
        NodeKind::Player(idx) => idx,
        NodeKind::Terminal => {
            return Err(
                "interactive request resolved to a terminal state; no decision available".to_string()
            )
        }
        NodeKind::Chance => {
            return Err("interactive request resolved to a chance node; expected player node".to_string())
        }
    };
    let hero = request.hero_seat.unwrap_or(actor);
    if hero != actor {
        return Err(format!(
            "hero_seat must match the current actor in this version (hero={hero}, actor={actor})"
        ));
    }
    Ok((state, hero))
}

fn run_interactive(config: &Config, model: &NlheGameModel, searcher: &RealtimeSearcher) -> AppResult<()> {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|err| format!("failed to read stdin: {err}"))?;
    if input.trim().is_empty() {
        return Err("interactive mode requires JSON input on stdin".to_string());
    }

    let request: InteractiveRequest =
        serde_json::from_str(&input).map_err(|err| format!("invalid interactive JSON input: {err}"))?;
    let (state, hero_seat) = build_state_from_interactive_request(&request, model)?;

    let mut search_cfg = build_search_config(config);
    if let Some(time_budget_ms) = request.time_budget_ms {
        search_cfg.time_budget_ms = time_budget_ms;
    }
    if let Some(threads) = request.threads {
        search_cfg.worker_threads = threads.max(1);
    }
    if let Some(max_iterations) = request.max_iterations {
        search_cfg.max_iterations = max_iterations;
    }
    if let Some(leaf_rollouts) = request.leaf_rollouts_per_action {
        search_cfg.leaf_rollouts_per_action = leaf_rollouts.max(1);
    }

    let result = searcher.search(model, &state, hero_seat, &search_cfg)?;
    let response = InteractiveResponse {
        mode: config.mode.as_str(),
        hero_seat,
        elapsed_ms: result.elapsed.as_millis(),
        iterations: result.iterations,
        chosen_action_index: result.chosen_action_index,
        chosen_action_token: result.chosen_action_token,
        action_probabilities: result
            .action_probabilities
            .into_iter()
            .map(|entry| InteractiveActionOut {
                action_index: entry.action_index,
                action_token: entry.action_token,
                policy_slot: entry.policy_slot,
                probability: entry.probability,
            })
            .collect(),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&response)
            .map_err(|err| format!("failed to serialize response JSON: {err}"))?
    );
    Ok(())
}

fn choose_action_for_scenario(
    action_space: &[cfr::nlhe_game::IndexedAction],
    rng: &mut StdRng,
) -> usize {
    if action_space.is_empty() {
        return 0;
    }
    let conservative = action_space
        .iter()
        .enumerate()
        .filter_map(|(idx, action)| {
            if action.sort_group <= 4 {
                Some(idx)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if !conservative.is_empty() && rng.gen::<f64>() < 0.85 {
        let choice = rng.gen_range(0..conservative.len());
        conservative[choice]
    } else {
        rng.gen_range(0..action_space.len())
    }
}

fn generate_benchmark_scenarios(
    model: &NlheGameModel,
    seed: u64,
    target_count: usize,
) -> Vec<BenchmarkScenario> {
    let mut rng = StdRng::seed_from_u64(seed ^ 0x13C6_D4A5_791B_2E07);
    let mut out = Vec::with_capacity(target_count);
    let target_steps = [0usize, 1, 2, 3, 4, 5, 7, 9, 12, 16];

    for (scenario_idx, step_target) in target_steps.iter().enumerate() {
        if out.len() >= target_count {
            break;
        }
        let mut candidate = None;
        for attempt in 0..96usize {
            let mut state = model.root_state();
            if matches!(model.node_kind(&state), NodeKind::Chance) {
                let deck_idx = (scenario_idx + attempt) % model.deck_seeds.len().max(1);
                state = model.next_state(&state, deck_idx);
            }
            let mut steps = 0usize;
            while steps < *step_target {
                match model.node_kind(&state) {
                    NodeKind::Terminal => break,
                    NodeKind::Chance => {
                        let probs = model.chance_probabilities(&state);
                        if probs.is_empty() {
                            break;
                        }
                        let idx = sample_from_probs(&probs, &mut rng);
                        state = model.next_state(&state, idx);
                    }
                    NodeKind::Player(player_idx) => {
                        let Some(game_state) = state.game.as_ref() else {
                            break;
                        };
                        let action_space = enumerate_action_space(game_state, player_idx);
                        if action_space.is_empty() {
                            break;
                        }
                        let action_idx = choose_action_for_scenario(&action_space, &mut rng);
                        state = model.next_state(&state, action_idx);
                        steps += 1;
                    }
                }
            }
            if let NodeKind::Player(hero_seat) = model.node_kind(&state) {
                candidate = Some(BenchmarkScenario {
                    name: format!(
                        "scenario_{:02}_{}",
                        scenario_idx + 1,
                        round_label(&state)
                    ),
                    state,
                    hero_seat,
                });
                break;
            }
        }
        if let Some(scenario) = candidate {
            out.push(scenario);
        }
    }

    while out.len() < target_count {
        let mut state = model.root_state();
        if matches!(model.node_kind(&state), NodeKind::Chance) {
            let deck_idx = out.len() % model.deck_seeds.len().max(1);
            state = model.next_state(&state, deck_idx);
        }
        if let NodeKind::Player(hero_seat) = model.node_kind(&state) {
            out.push(BenchmarkScenario {
                name: format!("fallback_{:02}_{}", out.len() + 1, round_label(&state)),
                state,
                hero_seat,
            });
        } else {
            break;
        }
    }
    out
}

fn run_benchmark(config: &Config, model: &NlheGameModel, searcher: &RealtimeSearcher) -> AppResult<()> {
    let scenarios = generate_benchmark_scenarios(model, config.seed, 10);
    if scenarios.is_empty() {
        return Err("failed to generate benchmark scenarios".to_string());
    }
    let scenario_count = scenarios.len();
    println!(
        "[rt-search] benchmark start: scenarios={} base_budget_ms={} threads={} gpu_batch={} adaptive_budget={}",
        scenarios.len(),
        config.time_budget_ms,
        config.threads,
        config.enable_gpu_batch,
        config.adaptive_budget
    );

    let mut total_elapsed_ms = 0.0f64;
    let mut total_iterations = 0usize;
    let mut total_infosets = 0usize;

    for scenario in scenarios {
        let game_state = scenario
            .state
            .game
            .as_ref()
            .ok_or_else(|| "benchmark scenario is missing resolved NLHE game state".to_string())?;
        let action_space = enumerate_action_space(game_state, scenario.hero_seat);
        let decision_budget_ms = resolve_budget_ms(config, &scenario.state, &action_space);
        let mut cfg = build_search_config(config);
        cfg.time_budget_ms = decision_budget_ms;
        let result = searcher.search(model, &scenario.state, scenario.hero_seat, &cfg)?;
        let chosen = result
            .action_probabilities
            .get(result.chosen_action_index)
            .map(|entry| (entry.action_token.clone(), entry.probability))
            .unwrap_or_else(|| ("unknown".to_string(), 0.0));
        let elapsed_ms = result.elapsed.as_millis() as f64;
        let iter_per_sec = if elapsed_ms > 0.0 {
            result.iterations as f64 * 1000.0 / elapsed_ms
        } else {
            0.0
        };
        let active_players = active_players_for_state(&scenario.state);
        total_elapsed_ms += elapsed_ms;
        total_iterations = total_iterations.saturating_add(result.iterations);
        total_infosets = total_infosets.saturating_add(result.infoset_count);
        println!(
            "[rt-search][benchmark] {} actor={} round={} active_players={} budget_ms={} elapsed_ms={} iterations={} iter_per_sec={:.1} infosets={} chosen={} p={:.3}",
            scenario.name,
            scenario.hero_seat,
            round_label(&scenario.state),
            active_players,
            decision_budget_ms,
            result.elapsed.as_millis(),
            result.iterations,
            iter_per_sec,
            result.infoset_count,
            chosen.0,
            chosen.1
        );
    }
    let avg_elapsed_ms = total_elapsed_ms / scenario_count as f64;
    let avg_iterations = total_iterations as f64 / scenario_count as f64;
    let avg_infosets = total_infosets as f64 / scenario_count as f64;
    println!(
        "[rt-search][benchmark] summary: avg_elapsed_ms={avg_elapsed_ms:.1} avg_iterations={avg_iterations:.1} avg_infosets={avg_infosets:.1}"
    );
    Ok(())
}

fn simulate_ring_hand_with_search(
    model: &NlheGameModel,
    searcher: &RealtimeSearcher,
    config: &Config,
    base_cfg: &RealtimeConfig,
    model_seat: usize,
    hand_seed: u64,
    stats: &mut RingDecisionStats,
) -> AppResult<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(hand_seed);
    let mut state = model.root_state();
    let mut opponent_kinds = Vec::with_capacity(model.config.num_players);
    let mut non_model_idx = 0usize;
    for seat in 0..model.config.num_players {
        if seat == model_seat {
            opponent_kinds.push(config.opponent);
            continue;
        }
        let kind = config
            .opponents
            .get(non_model_idx)
            .copied()
            .unwrap_or(config.opponent);
        opponent_kinds.push(kind);
        non_model_idx += 1;
    }
    let mut opponent_policies = opponent_kinds
        .into_iter()
        .map(OpponentPolicy::from_kind)
        .collect::<Vec<_>>();

    loop {
        match model.node_kind(&state) {
            NodeKind::Terminal => {
                let utilities = (0..model.config.num_players)
                    .map(|seat| model.terminal_utility(&state, seat))
                    .collect::<Vec<_>>();
                return Ok(utilities);
            }
            NodeKind::Chance => {
                let probs = model.chance_probabilities(&state);
                let action_idx = sample_from_probs(&probs, &mut rng);
                state = model.next_state(&state, action_idx);
            }
            NodeKind::Player(player_idx) => {
                let game_state = state
                    .game
                    .as_ref()
                    .ok_or_else(|| "expected resolved NLHE state at ring-eval player node".to_string())?;
                let action_space = enumerate_action_space(game_state, player_idx);
                if action_space.is_empty() {
                    let utilities = (0..model.config.num_players)
                        .map(|seat| model.terminal_utility(&state, seat))
                        .collect::<Vec<_>>();
                    return Ok(utilities);
                }

                let action_idx = if player_idx == model_seat {
                    let decision_budget_ms = resolve_budget_ms(config, &state, &action_space);
                    let mut search_cfg = base_cfg.clone();
                    search_cfg.time_budget_ms = decision_budget_ms;
                    let result = searcher.search(model, &state, player_idx, &search_cfg)?;
                    stats.decisions = stats.decisions.saturating_add(1);
                    stats.total_budget_ms = stats.total_budget_ms.saturating_add(decision_budget_ms);
                    stats.total_search_ms = stats
                        .total_search_ms
                        .saturating_add(result.elapsed.as_millis());
                    stats.total_iterations = stats
                        .total_iterations
                        .saturating_add(result.iterations as u64);
                    stats.total_infosets = stats
                        .total_infosets
                        .saturating_add(result.infoset_count as u64);
                    sample_action_from_result(&result, &mut rng)
                } else {
                    let strategy = opponent_policies[player_idx].get_strategy(&state, player_idx)?;
                    if strategy.len() != action_space.len() {
                        return Err(format!(
                            "opponent strategy/action mismatch for seat {player_idx}: strategy={}, actions={}",
                            strategy.len(),
                            action_space.len()
                        ));
                    }
                    sample_from_probs(&strategy, &mut rng)
                };
                let action_idx = action_idx.min(action_space.len().saturating_sub(1));
                state = model.next_state(&state, action_idx);
            }
        }
    }
}

fn run_ring_eval(config: &Config, model: &NlheGameModel, searcher: &RealtimeSearcher) -> AppResult<()> {
    let cfg = RealtimeConfig {
        time_budget_ms: config.time_budget_ms,
        worker_threads: config.threads,
        max_iterations: config.max_iterations,
        leaf_rollouts_per_action: config.leaf_rollouts,
        enable_gpu_batch: config.enable_gpu_batch,
        rng_seed: config.seed ^ 0x44E7_1C2A_A15E_11BF,
    };
    println!(
        "[rt-search] ring-eval start: opponent={} opponents=[{}] hands={} model_seat={} base_budget_ms={} threads={} gpu_batch={} adaptive_budget={}",
        config.opponent.as_str(),
        format_opponents(&config.opponents),
        config.hands,
        config.model_seat,
        cfg.time_budget_ms,
        cfg.worker_threads,
        cfg.enable_gpu_batch,
        config.adaptive_budget
    );
    let done = AtomicUsize::new(0);
    let started = Instant::now();

    let mut report = RingEvalReport {
        hands: config.hands,
        model_total_bb: 0.0,
        per_seat_total_bb: vec![0.0; config.num_players],
        model_wins: 0,
        model_losses: 0,
        ties: 0,
    };
    let mut decision_stats = RingDecisionStats::default();

    for hand_idx in 0..config.hands {
        let hand_seed = config.seed ^ ((hand_idx as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03));
        let utilities = simulate_ring_hand_with_search(
            model,
            searcher,
            config,
            &cfg,
            config.model_seat,
            hand_seed,
            &mut decision_stats,
        )?;
        if utilities.len() != config.num_players {
            return Err(format!(
                "terminal utility size mismatch: expected {}, got {}",
                config.num_players,
                utilities.len()
            ));
        }
        for (seat, value) in utilities.iter().enumerate() {
            report.per_seat_total_bb[seat] += *value;
        }
        let model_value = utilities[config.model_seat];
        report.model_total_bb += model_value;
        if model_value > 1e-12 {
            report.model_wins += 1;
        } else if model_value < -1e-12 {
            report.model_losses += 1;
        } else {
            report.ties += 1;
        }

        let completed = done.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        if config.progress_every > 0 && (completed % config.progress_every == 0 || completed == config.hands)
        {
            let elapsed = started.elapsed().as_secs_f64();
            let done_clamped = completed.min(config.hands);
            let pct = (done_clamped as f64) * 100.0 / config.hands as f64;
            let eta = if done_clamped > 0 && done_clamped < config.hands {
                elapsed * (config.hands - done_clamped) as f64 / done_clamped as f64
            } else {
                0.0
            };
            let hps = if elapsed > 0.0 {
                done_clamped as f64 / elapsed
            } else {
                0.0
            };
            let avg_decision_ms = if decision_stats.decisions > 0 {
                decision_stats.total_search_ms as f64 / decision_stats.decisions as f64
            } else {
                0.0
            };
            println!(
                "[rt-search][ring-eval] progress: {done_clamped}/{} ({pct:.1}%) elapsed={} eta~{} speed={hps:.2} hands/s decisions={} avg_decision_ms={avg_decision_ms:.1}",
                config.hands,
                format_duration(elapsed),
                format_duration(eta),
                decision_stats.decisions
            );
        }
    }

    let elapsed = started.elapsed().as_secs_f64();
    let model_bb_per_hand = report.model_total_bb / report.hands as f64;
    let model_bb_per_100 = model_bb_per_hand * 100.0;
    let per_seat_bb_per_100 = report
        .per_seat_total_bb
        .iter()
        .map(|value| *value / report.hands as f64 * 100.0)
        .collect::<Vec<_>>();
    let zero_sum_check_bb_per_hand =
        report.per_seat_total_bb.iter().sum::<f64>() / report.hands as f64;
    let average_decision_ms = if decision_stats.decisions > 0 {
        decision_stats.total_search_ms as f64 / decision_stats.decisions as f64
    } else {
        0.0
    };
    let average_iterations_per_decision = if decision_stats.decisions > 0 {
        decision_stats.total_iterations as f64 / decision_stats.decisions as f64
    } else {
        0.0
    };
    let average_infosets_per_decision = if decision_stats.decisions > 0 {
        decision_stats.total_infosets as f64 / decision_stats.decisions as f64
    } else {
        0.0
    };
    let average_budget_ms = if decision_stats.decisions > 0 {
        decision_stats.total_budget_ms as f64 / decision_stats.decisions as f64
    } else {
        0.0
    };
    let hands_per_second = if elapsed > 0.0 {
        report.hands as f64 / elapsed
    } else {
        0.0
    };

    println!(
        "[rt-search][ring-eval] done: hands={} elapsed={elapsed:.1}s speed={hands_per_second:.3} hands/s bb_per_100={model_bb_per_100:+.3} wins={} losses={} ties={} decisions={} avg_decision_ms={average_decision_ms:.1} avg_iters={average_iterations_per_decision:.1} avg_infosets={average_infosets_per_decision:.1} avg_budget_ms={average_budget_ms:.1}",
        report.hands,
        report.model_wins,
        report.model_losses,
        report.ties,
        decision_stats.decisions
    );
    let payload = serde_json::json!({
        "mode": "ring-eval",
        "opponent": config.opponent.as_str(),
        "opponents": config.opponents.iter().map(|item| item.as_str()).collect::<Vec<_>>(),
        "hands": report.hands,
        "model_seat": config.model_seat,
        "model_bb_per_hand": model_bb_per_hand,
        "model_bb_per_100": model_bb_per_100,
        "per_seat_bb_per_100": per_seat_bb_per_100,
        "model_wins": report.model_wins,
        "model_losses": report.model_losses,
        "ties": report.ties,
        "elapsed_sec": elapsed,
        "hands_per_second": hands_per_second,
        "zero_sum_check_bb_per_hand": zero_sum_check_bb_per_hand,
        "decisions": decision_stats.decisions,
        "average_decision_ms": average_decision_ms,
        "average_iterations_per_decision": average_iterations_per_decision,
        "average_infosets_per_decision": average_infosets_per_decision,
        "average_budget_ms": average_budget_ms
    });
    println!(
        "REALTIME_RING_EVAL_JSON {}",
        serde_json::to_string(&payload).map_err(|err| format!("failed to encode ring-eval json: {err}"))?
    );
    Ok(())
}

// ── Live mode ────────────────────────────────────────────────────────────────
// Persistent JSON-line protocol: one JSON object per line on stdin, one JSON
// object per line on stdout.  Used by the Python vision bridge.

#[derive(Debug, Deserialize)]
struct LiveAction {
    seat: usize,
    #[serde(rename = "type")]
    action_type: String,
    total: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct LiveRequest {
    hero_seat: usize,
    hero_cards: Vec<String>,
    board: Vec<String>,
    num_players: Option<usize>,
    dealer_seat: Option<usize>,
    stacks: Option<Vec<u32>>,
    actions: Option<Vec<LiveAction>>,
    time_budget_ms: Option<u64>,
    samples: Option<usize>,
}

#[derive(Debug, Serialize)]
struct LiveActionOut {
    action: String,
    live_action: String,
    amount: u32,
    probability: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
struct ReplayAlignmentCounts {
    requested_actions: usize,
    applied_actions: usize,
    dropped_actions: usize,
    auto_alignment_steps: usize,
    actor_mismatch_actions: usize,
    guard_exhausted_actions: usize,
    replay_chance_steps: usize,
    post_replay_chance_steps: usize,
}

#[derive(Debug, Clone, Default, Serialize)]
struct ReplayAlignmentSummary {
    requested_actions: usize,
    build_samples: usize,
    total_applied_actions: usize,
    total_dropped_actions: usize,
    total_auto_alignment_steps: usize,
    total_actor_mismatch_actions: usize,
    total_guard_exhausted_actions: usize,
    total_replay_chance_steps: usize,
    total_post_replay_chance_steps: usize,
}

impl ReplayAlignmentSummary {
    fn with_requested_actions(requested_actions: usize) -> Self {
        Self {
            requested_actions,
            ..Self::default()
        }
    }

    fn observe(&mut self, counts: &ReplayAlignmentCounts) {
        self.build_samples = self.build_samples.saturating_add(1);
        self.total_applied_actions = self
            .total_applied_actions
            .saturating_add(counts.applied_actions);
        self.total_dropped_actions = self
            .total_dropped_actions
            .saturating_add(counts.dropped_actions);
        self.total_auto_alignment_steps = self
            .total_auto_alignment_steps
            .saturating_add(counts.auto_alignment_steps);
        self.total_actor_mismatch_actions = self
            .total_actor_mismatch_actions
            .saturating_add(counts.actor_mismatch_actions);
        self.total_guard_exhausted_actions = self
            .total_guard_exhausted_actions
            .saturating_add(counts.guard_exhausted_actions);
        self.total_replay_chance_steps = self
            .total_replay_chance_steps
            .saturating_add(counts.replay_chance_steps);
        self.total_post_replay_chance_steps = self
            .total_post_replay_chance_steps
            .saturating_add(counts.post_replay_chance_steps);
    }
}

#[derive(Debug, Serialize)]
struct LiveResponse {
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    hero_seat: usize,
    recommended_token: String,
    recommended_action: String,
    recommended_amount: u32,
    action_probabilities: Vec<LiveActionOut>,
    elapsed_ms: u128,
    iterations: usize,
    samples_averaged: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    replay_alignment: Option<ReplayAlignmentSummary>,
}

/// Build a deck arranged so that known cards appear at the positions the engine
/// expects:  seats 0..n get 2 cards each (indices 0..2n), then burn+flop at
/// 2n, 2n+1, 2n+2, 2n+3 (burn + 3 flop), then 2n+4=burn, 2n+5=turn,
/// 2n+6=burn, 2n+7=river.  Unknown positions are filled from the remaining
/// shuffled deck.
fn build_live_deck(
    num_players: usize,
    hero_seat: usize,
    hero_cards: &[Card],
    board: &[Card],
    rng: &mut StdRng,
) -> Deck {
    use std::collections::HashSet;

    let known_set: HashSet<u64> = hero_cards
        .iter()
        .chain(board.iter())
        .map(|c| card_sort_key(*c))
        .collect();

    // Full 52-card ordered deck, shuffle the unknowns.
    let all_cards = all_52_cards();
    let mut unknowns: Vec<Card> = all_cards
        .into_iter()
        .filter(|c| !known_set.contains(&card_sort_key(*c)))
        .collect();
    // Shuffle unknown cards
    use rand::seq::SliceRandom;
    unknowns.shuffle(rng);

    let mut arranged: Vec<Card> = Vec::with_capacity(52);
    let mut unknown_iter = unknowns.into_iter();

    // Seat hole cards: 2 cards each, hero's seat gets hero_cards, others get randoms.
    for seat in 0..num_players {
        if seat == hero_seat {
            arranged.push(hero_cards[0]);
            arranged.push(hero_cards[1]);
        } else {
            arranged.push(unknown_iter.next().unwrap_or(Card::parse("2c").unwrap()));
            arranged.push(unknown_iter.next().unwrap_or(Card::parse("3c").unwrap()));
        }
    }

    // Burn card before flop.
    arranged.push(unknown_iter.next().unwrap_or(Card::parse("4c").unwrap()));
    // Flop (3 cards).
    for i in 0..3 {
        if i < board.len() {
            arranged.push(board[i]);
        } else {
            arranged.push(unknown_iter.next().unwrap_or(Card::parse("5c").unwrap()));
        }
    }
    // Burn before turn.
    arranged.push(unknown_iter.next().unwrap_or(Card::parse("6c").unwrap()));
    // Turn.
    if board.len() >= 4 {
        arranged.push(board[3]);
    } else {
        arranged.push(unknown_iter.next().unwrap_or(Card::parse("7c").unwrap()));
    }
    // Burn before river.
    arranged.push(unknown_iter.next().unwrap_or(Card::parse("8c").unwrap()));
    // River.
    if board.len() >= 5 {
        arranged.push(board[4]);
    } else {
        arranged.push(unknown_iter.next().unwrap_or(Card::parse("9c").unwrap()));
    }
    // Fill remainder.
    for c in unknown_iter {
        arranged.push(c);
    }

    Deck::from_arranged(arranged)
}

fn card_sort_key(c: Card) -> u64 {
    let rank_idx = c.rank as u64;
    let suit_idx = c.suit as u64;
    rank_idx * 4 + suit_idx
}

fn all_52_cards() -> Vec<Card> {
    let ranks = [
        Rank::Two, Rank::Three, Rank::Four, Rank::Five, Rank::Six, Rank::Seven,
        Rank::Eight, Rank::Nine, Rank::Ten, Rank::Jack, Rank::Queen, Rank::King, Rank::Ace,
    ];
    let suits = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];
    let mut cards = Vec::with_capacity(52);
    for &rank in &ranks {
        for &suit in &suits {
            cards.push(Card { rank, suit });
        }
    }
    cards
}

fn rotate_live_seat_to_engine(
    seat: usize,
    dealer_seat: Option<usize>,
    num_players: usize,
) -> usize {
    if num_players == 0 {
        return seat;
    }
    let Some(dealer) = dealer_seat.map(|value| value % num_players) else {
        return seat % num_players;
    };
    let small_blind_live = (dealer + 1) % num_players;
    (seat + num_players - small_blind_live) % num_players
}

/// Reconstruct NlheState for the live request by replaying the action history
/// against a freshly built game with the known deck arrangement.
fn build_live_state(
    req: &LiveRequest,
    config: &Config,
    model: &NlheGameModel,
    rng: &mut StdRng,
) -> AppResult<(NlheState, usize, ReplayAlignmentCounts)> {
    let num_players = req.num_players.unwrap_or(config.num_players);
    let dealer_seat = req.dealer_seat.map(|seat| seat % num_players);
    let hero_seat = rotate_live_seat_to_engine(req.hero_seat, dealer_seat, num_players);

    let hero_cards: Vec<Card> = req
        .hero_cards
        .iter()
        .filter_map(|s| Card::parse(s))
        .collect();
    if hero_cards.len() != 2 {
        return Err(format!(
            "hero_cards must contain exactly 2 valid cards, got {:?}",
            req.hero_cards
        ));
    }

    let board: Vec<Card> = req
        .board
        .iter()
        .filter_map(|s| Card::parse(s))
        .collect();

    let deck = build_live_deck(num_players, hero_seat, &hero_cards, &board, rng);

    let nlhe_config = NlheConfig {
        num_players,
        starting_stack: config.starting_stack,
        small_blind: config.small_blind,
        big_blind: config.big_blind,
    };

    let mut game = NlheGame::new_with_deck(nlhe_config, deck)
        .map_err(|err| format!("failed to build live game: {err:?}"))?;

    // Wrap game in NlheState with empty history (preflop start).
    let mut state = NlheState {
        game: Some(game),
        pending_chance: false,
        action_history: Vec::new(),
        action_actors: Vec::new(),
    };

    // Replay action history to advance state to current position.
    let actions = req.actions.as_deref().unwrap_or(&[]);
    let mut replay_alignment = ReplayAlignmentCounts {
        requested_actions: actions.len(),
        ..ReplayAlignmentCounts::default()
    };
    for live_action in actions {
        let target_actor = rotate_live_seat_to_engine(live_action.seat, dealer_seat, num_players);
        let observed_type = live_action.action_type.to_ascii_lowercase();
        let should_align = matches!(
            observed_type.as_str(),
            "bet" | "raise" | "allin" | "all-in" | "all_in" | "ai"
        );
        let mut align_guard = 0usize;
        let max_align_steps = num_players.saturating_mul(4).max(8);
        let mut action_applied = false;
        let mut guard_exhausted = false;
        loop {
            match model.node_kind(&state) {
                NodeKind::Terminal => break,
                NodeKind::Chance => {
                    state = model.next_state(&state, 0);
                    replay_alignment.replay_chance_steps =
                        replay_alignment.replay_chance_steps.saturating_add(1);
                    align_guard += 1;
                    if align_guard >= max_align_steps {
                        guard_exhausted = true;
                        break;
                    }
                }
                NodeKind::Player(actor) => {
                    let game_ref = state
                        .game
                        .as_ref()
                        .ok_or_else(|| "expected game state during action replay".to_string())?;
                    let action_space = enumerate_action_space(game_ref, actor);
                    if action_space.is_empty() {
                        break;
                    }
                    if should_align && actor != target_actor && align_guard < max_align_steps {
                        if let Some(auto_idx) = pick_alignment_action_index(&action_space) {
                            state = model.next_state(&state, auto_idx);
                            replay_alignment.auto_alignment_steps =
                                replay_alignment.auto_alignment_steps.saturating_add(1);
                            align_guard += 1;
                            continue;
                        }
                    }
                    if should_align && actor != target_actor {
                        replay_alignment.actor_mismatch_actions =
                            replay_alignment.actor_mismatch_actions.saturating_add(1);
                    }
                    // Apply the observed live action on the current actor. If
                    // actor-seat alignment could not be achieved (e.g. seat is
                    // already folded), this keeps compatibility with older
                    // behavior while still benefiting from alignment when
                    // possible.
                    let chosen_idx = map_live_action_to_index(live_action, &action_space, config)?;
                    state = model.next_state(&state, chosen_idx);
                    action_applied = true;
                    break;
                }
            }
            if align_guard >= max_align_steps {
                guard_exhausted = true;
                break;
            }
        }
        if action_applied {
            replay_alignment.applied_actions = replay_alignment.applied_actions.saturating_add(1);
        } else if guard_exhausted {
            replay_alignment.guard_exhausted_actions =
                replay_alignment.guard_exhausted_actions.saturating_add(1);
        }
    }
    replay_alignment.dropped_actions = replay_alignment
        .requested_actions
        .saturating_sub(replay_alignment.applied_actions);

    // If action replay ends exactly on a chance node (for example right after a
    // street transition), advance deterministic chance nodes so the live query
    // can evaluate the next player node.
    loop {
        match model.node_kind(&state) {
            NodeKind::Chance => {
                state = model.next_state(&state, 0);
                replay_alignment.post_replay_chance_steps = replay_alignment
                    .post_replay_chance_steps
                    .saturating_add(1);
            }
            _ => break,
        }
    }

    // Apply live remaining stacks after action replay so the reconstructed
    // legal action space reflects the visible end-of-sequence state rather
    // than double-deducting chips during replay.
    if let Some(stacks) = req.stacks.as_ref() {
        if let Some(game_state) = state.game.as_mut() {
            for (live_seat, &chip_count) in stacks.iter().enumerate() {
                if live_seat >= num_players || chip_count == 0 {
                    continue;
                }
                let engine_seat = rotate_live_seat_to_engine(live_seat, dealer_seat, num_players);
                if engine_seat < game_state.players.len() {
                    game_state.players[engine_seat].stack = chip_count;
                }
            }
        }
    }

    Ok((state, hero_seat, replay_alignment))
}

fn pick_alignment_action_index(action_space: &[IndexedAction]) -> Option<usize> {
    find_action_index(action_space, |a| a.action_token == "x")
        .or_else(|| find_action_index(action_space, |a| a.action_token == "c"))
        .or_else(|| find_action_index(action_space, |a| a.action_token == "f"))
}

fn map_live_action_to_index(
    live_action: &LiveAction,
    action_space: &[IndexedAction],
    _config: &Config,
) -> AppResult<usize> {
    let action_type = live_action.action_type.to_ascii_lowercase();
    let total = live_action.total.unwrap_or(0);

    match action_type.as_str() {
        "fold" | "f" => {
            find_action_index(action_space, |a| a.action_token == "f")
                .ok_or_else(|| "no fold action available".to_string())
        }
        "check" | "x" => {
            find_action_index(action_space, |a| a.action_token == "x")
                .ok_or_else(|| "no check action available".to_string())
        }
        "call" | "c" => {
            find_action_index(action_space, |a| a.action_token == "c")
                .ok_or_else(|| "no call action available".to_string())
        }
        "allin" | "all-in" | "all_in" | "ai" => {
            find_action_index(action_space, |a| a.action_token == "ai")
                .or_else(|| {
                    // Fall back to largest bet/raise available.
                    action_space
                        .iter()
                        .enumerate()
                        .max_by_key(|(_, a)| a.sort_amount)
                        .map(|(i, _)| i)
                })
                .ok_or_else(|| "no all-in action available".to_string())
        }
        "bet" | "raise" => {
            // Find the abstract action whose sort_amount is closest to total.
            if total == 0 {
                // Pick smallest bet/raise slot.
                find_action_index(action_space, |a| a.sort_group >= 3)
                    .ok_or_else(|| "no bet/raise action available".to_string())
            } else {
                action_space
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| a.sort_group >= 3)
                    .min_by_key(|(_, a)| (a.sort_amount as i64 - total as i64).abs())
                    .map(|(i, _)| i)
                    .ok_or_else(|| "no bet/raise action available".to_string())
            }
        }
        _ => Err(format!(
            "unrecognized live action type: {}",
            live_action.action_type
        )),
    }
}

fn find_action_index<F>(action_space: &[IndexedAction], predicate: F) -> Option<usize>
where
    F: Fn(&IndexedAction) -> bool,
{
    action_space.iter().position(predicate)
}

fn best_action_and_amount(
    action_probs: &[(String, f64, u8)],
    action_amounts: &HashMap<String, u32>,
) -> (String, String, u32) {
    let best = action_probs
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    match best {
        Some((token, _, _)) => {
            let amount = action_amounts
                .get(token)
                .copied()
                .unwrap_or_else(|| extract_amount_from_token(token));
            (token.clone(), token_to_live_action(token), amount)
        }
        None => (String::new(), "fold".to_string(), 0),
    }
}

fn token_to_live_action(token: &str) -> String {
    let token = token.to_ascii_lowercase();
    match token.as_str() {
        "f" => "fold".to_string(),
        "x" => "check".to_string(),
        "c" => "call".to_string(),
        "ai" => "allin".to_string(),
        _ if token.starts_with("b:") => "bet".to_string(),
        _ if token.starts_with("r:") => "raise".to_string(),
        _ => token,
    }
}

fn extract_amount_from_token(token: &str) -> u32 {
    // Tokens like "b:bet_0.33_pot", "r:3bet_2.50x", "c", "f", "x", "ai"
    // Extract trailing number if present.
    let mut num = String::new();
    let mut has_dot = false;
    for ch in token.chars().rev() {
        if ch.is_ascii_digit() {
            num.insert(0, ch);
        } else if ch == '.' && !has_dot {
            num.insert(0, ch);
            has_dot = true;
        } else if !num.is_empty() {
            break;
        }
    }
    num.parse::<f64>().map(|v| v as u32).unwrap_or(0)
}

fn run_live(config: &Config, model: &NlheGameModel, searcher: &RealtimeSearcher) -> AppResult<()> {
    eprintln!("[rt-search] live mode ready — waiting for JSON-line requests on stdin");
    io::stdout()
        .write_all(b"{\"status\":\"ready\"}\n")
        .map_err(|err| format!("stdout write error: {err}"))?;
    io::stdout().flush().ok();

    let stdin = io::stdin();
    let reader = BufReader::new(stdin.lock());

    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let response = process_live_line(&line, config, model, searcher);
        let json_out = serde_json::to_string(&response)
            .unwrap_or_else(|_| r#"{"status":"error","error":"serialization failed","hero_seat":0,"recommended_action":"fold","recommended_amount":0,"action_probabilities":[],"elapsed_ms":0,"iterations":0,"samples_averaged":0}"#.to_string());
        println!("{json_out}");
        io::stdout().flush().ok();
    }
    Ok(())
}

fn process_live_line(
    line: &str,
    config: &Config,
    model: &NlheGameModel,
    searcher: &RealtimeSearcher,
) -> LiveResponse {
    let req: LiveRequest = match serde_json::from_str(line) {
        Ok(r) => r,
        Err(err) => {
            return LiveResponse {
                status: "error",
                error: Some(format!("JSON parse error: {err}")),
                hero_seat: 0,
                recommended_token: String::new(),
                recommended_action: "fold".to_string(),
                recommended_amount: 0,
                action_probabilities: vec![],
                elapsed_ms: 0,
                iterations: 0,
                samples_averaged: 0,
                replay_alignment: None,
            };
        }
    };

    let num_samples = req.samples.unwrap_or(4).max(1).min(16);
    let time_budget_ms = req.time_budget_ms.unwrap_or(config.time_budget_ms);
    let per_sample_budget = (time_budget_ms / num_samples as u64).max(500);

    let mut rng = StdRng::seed_from_u64(config.seed ^ 0xDEAD_BEEF_1234_5678);
    let mut total_probs: Vec<(String, f64, u8)> = Vec::new();
    let mut action_amounts: HashMap<String, u32> = HashMap::new();
    let mut total_iterations = 0usize;
    let mut valid_samples = 0usize;
    let requested_actions = req.actions.as_ref().map(|rows| rows.len()).unwrap_or(0);
    let mut replay_alignment = ReplayAlignmentSummary::with_requested_actions(requested_actions);
    let start = Instant::now();

    for _ in 0..num_samples {
        let sample_seed = rng.gen::<u64>();
        let mut sample_rng = StdRng::seed_from_u64(sample_seed);

        let (state, hero_seat, alignment_counts) = match build_live_state(&req, config, model, &mut sample_rng) {
            Ok(v) => v,
            Err(_) => continue,
        };
        replay_alignment.observe(&alignment_counts);

        // Verify hero's turn.
        if !matches!(model.node_kind(&state), NodeKind::Player(p) if p == hero_seat) {
            // Not hero's turn in this sample; try to still get blueprint probabilities.
            continue;
        }

        if let Some(game_state) = state.game.as_ref() {
            for action in enumerate_action_space(game_state, hero_seat) {
                action_amounts
                    .entry(action.action_token.clone())
                    .or_insert(action.sort_amount);
            }
        }

        let mut search_cfg = build_search_config(config);
        search_cfg.time_budget_ms = per_sample_budget;

        let result = match searcher.search(model, &state, hero_seat, &search_cfg) {
            Ok(r) => r,
            Err(_) => continue,
        };

        total_iterations += result.iterations;
        valid_samples += 1;

        for entry in &result.action_probabilities {
            if let Some(existing) = total_probs.iter_mut().find(|(t, _, _)| t == &entry.action_token) {
                existing.1 += entry.probability;
            } else {
                total_probs.push((entry.action_token.clone(), entry.probability, entry.policy_slot as u8));
            }
        }
    }

    let elapsed_ms = start.elapsed().as_millis();

    if valid_samples == 0 || total_probs.is_empty() {
        return LiveResponse {
            status: "error",
            error: Some("all samples failed or state is not hero's turn".to_string()),
            hero_seat: req.hero_seat,
            recommended_token: String::new(),
            recommended_action: "fold".to_string(),
            recommended_amount: 0,
            action_probabilities: vec![],
            elapsed_ms,
            iterations: total_iterations,
            samples_averaged: 0,
            replay_alignment: Some(replay_alignment),
        };
    }

    // Normalize probabilities across samples.
    let denom = valid_samples as f64;
    for entry in &mut total_probs {
        entry.1 /= denom;
    }
    total_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let (recommended_token, recommended_action, recommended_amount) =
        best_action_and_amount(&total_probs, &action_amounts);

    let action_probabilities = total_probs
        .into_iter()
        .map(|(action, probability, _)| {
            let amount = action_amounts
                .get(&action)
                .copied()
                .unwrap_or_else(|| extract_amount_from_token(&action));
            let live_action = token_to_live_action(&action);
            LiveActionOut {
                action,
                live_action,
                amount,
                probability,
            }
        })
        .collect();

    LiveResponse {
        status: "ok",
        error: None,
        hero_seat: req.hero_seat,
        recommended_token,
        recommended_action,
        recommended_amount,
        action_probabilities,
        elapsed_ms,
        iterations: total_iterations,
        samples_averaged: valid_samples,
        replay_alignment: Some(replay_alignment),
    }
}

fn run_main() -> AppResult<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_help();
        return Ok(());
    }

    let config = parse_config(&args)?;
    println!(
        "[rt-search] mode={} opponent={} opponents=[{}] model={} policy={} players={} stack={} blinds={}/{} deck_samples={} gpu_batch={} adaptive_budget={} batch_size={} batch_wait_us={}",
        config.mode.as_str(),
        config.opponent.as_str(),
        format_opponents(&config.opponents),
        config.model_path.display(),
        config.policy_output.as_str(),
        config.num_players,
        config.starting_stack,
        config.small_blind,
        config.big_blind,
        config.deck_samples,
        config.enable_gpu_batch,
        config.adaptive_budget,
        config.batch_size,
        config.batch_wait_us
    );
    let model = build_game_model(&config)?;
    let batch_runtime = BatchRuntimeConfig {
        max_batch_size: config.batch_size,
        max_wait_us: config.batch_wait_us,
        queue_capacity: config.batch_queue_capacity,
        use_cuda: true,
        cuda_device_id: 0,
        cuda_tf32: true,
    };
    let searcher = RealtimeSearcher::from_model(&config.model_path)
        .with_output_mode(config.policy_output)
        .with_batch_runtime(batch_runtime);

    match config.mode {
        Mode::Interactive => run_interactive(&config, &model, &searcher)?,
        Mode::Benchmark => run_benchmark(&config, &model, &searcher)?,
        Mode::RingEval => run_ring_eval(&config, &model, &searcher)?,
        Mode::Live => run_live(&config, &model, &searcher)?,
    }
    Ok(())
}

fn main() {
    let builder = std::thread::Builder::new()
        .name("deep-cfr-realtime-play-main".to_string())
        .stack_size(64 * 1024 * 1024);
    let handle = builder
        .spawn(run_main)
        .unwrap_or_else(|err| panic!("failed to spawn realtime_play thread: {err}"));

    match handle.join() {
        Ok(Ok(())) => {}
        Ok(Err(err)) => {
            eprintln!("[rt-search] error: {err}");
            std::process::exit(1);
        }
        Err(_) => panic!("realtime_play thread panicked"),
    }
}
