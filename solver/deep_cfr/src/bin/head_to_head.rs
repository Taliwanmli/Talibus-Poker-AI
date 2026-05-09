use cfr::external_sampling::{GameModel, NodeKind};
use cfr::nlhe_game::{enumerate_action_space, NlheGameModel, NlheState};
use deep_cfr::encoding::encode_nlhe_state;
use deep_cfr::onnx_policy::{OnnxPolicy, PolicyOutput};
use deep_cfr::sample::MAX_ACTIONS;
use deep_cfr::tag_policy::TagPolicy;
use game::{BettingRound, NlheConfig};
use rand::prelude::{Rng, SeedableRng, StdRng};
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_CLUSTER_DIR: &str = "checkpoints/nlhe_clusters";
const DEFAULT_HANDS: usize = 10_000;
const DEFAULT_DECK_SAMPLES: usize = 10_000;
const DEFAULT_WORKERS: usize = 0;
const DEFAULT_PROGRESS_EVERY: usize = 1_000;
const DEFAULT_STARTING_STACK: u32 = 2_000;
const DEFAULT_SMALL_BLIND: u32 = 10;
const DEFAULT_BIG_BLIND: u32 = 20;
const DEBUG_LOG_FILE: &str = "debug-1270ea.log";
const DEBUG_SESSION_ID: &str = "1270ea";
const DEBUG_RUN_ID: &str = "tag_eval_debug";
const DEBUG_ACTION_GROUPS: usize = 6;
const DEBUG_ROUNDS: usize = 4;
const DEBUG_PRESSURE_BUCKETS: usize = 4;
const DEBUG_RESULT_BUCKETS: usize = 3;
const DEBUG_PRE_TIER_COUNT: usize = 5;

type AppResult<T> = Result<T, String>;

#[derive(Clone, Debug, Default)]
struct DebugStats {
    model_a_win_bucket_counts: [u64; DEBUG_RESULT_BUCKETS],
    model_a_loss_bucket_counts: [u64; DEBUG_RESULT_BUCKETS],
    model_a_win_total_bb: f64,
    model_a_loss_total_abs_bb: f64,
    action_counts: [[[[u64; DEBUG_ACTION_GROUPS]; DEBUG_PRESSURE_BUCKETS]; DEBUG_ROUNDS]; 2],
    model_a_preflop_aggro_hands_by_tier: [u64; DEBUG_PRE_TIER_COUNT],
    model_a_preflop_allin_hands_by_tier: [u64; DEBUG_PRE_TIER_COUNT],
    model_a_big_loss_preflop_by_tier: [u64; DEBUG_PRE_TIER_COUNT],
    model_a_big_win_preflop_by_tier: [u64; DEBUG_PRE_TIER_COUNT],
    top_losses: Vec<HandTrace>,
    top_wins: Vec<HandTrace>,
}

impl DebugStats {
    fn merge(&mut self, other: Self) {
        for idx in 0..DEBUG_RESULT_BUCKETS {
            self.model_a_win_bucket_counts[idx] += other.model_a_win_bucket_counts[idx];
            self.model_a_loss_bucket_counts[idx] += other.model_a_loss_bucket_counts[idx];
        }
        for idx in 0..DEBUG_PRE_TIER_COUNT {
            self.model_a_preflop_aggro_hands_by_tier[idx] +=
                other.model_a_preflop_aggro_hands_by_tier[idx];
            self.model_a_preflop_allin_hands_by_tier[idx] +=
                other.model_a_preflop_allin_hands_by_tier[idx];
            self.model_a_big_loss_preflop_by_tier[idx] +=
                other.model_a_big_loss_preflop_by_tier[idx];
            self.model_a_big_win_preflop_by_tier[idx] += other.model_a_big_win_preflop_by_tier[idx];
        }
        self.model_a_win_total_bb += other.model_a_win_total_bb;
        self.model_a_loss_total_abs_bb += other.model_a_loss_total_abs_bb;
        for side in 0..2 {
            for round in 0..DEBUG_ROUNDS {
                for pressure in 0..DEBUG_PRESSURE_BUCKETS {
                    for group in 0..DEBUG_ACTION_GROUPS {
                        self.action_counts[side][round][pressure][group] +=
                            other.action_counts[side][round][pressure][group];
                    }
                }
            }
        }
        self.top_losses.extend(other.top_losses);
        self.top_losses.sort_by(|a, b| {
            a.model_a_bb
                .partial_cmp(&b.model_a_bb)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.top_losses.truncate(8);

        self.top_wins.extend(other.top_wins);
        self.top_wins.sort_by(|a, b| {
            b.model_a_bb
                .partial_cmp(&a.model_a_bb)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.top_wins.truncate(8);
    }

    fn record_result(&mut self, trace: HandTrace) {
        let model_a_bb = trace.model_a_bb;
        let tier_idx = trace.model_a_preflop_tier.min(DEBUG_PRE_TIER_COUNT - 1);
        if trace.max_round == 0 && (trace.model_a_actions[4] > 0 || trace.model_a_actions[5] > 0) {
            self.model_a_preflop_aggro_hands_by_tier[tier_idx] += 1;
        }
        if trace.max_round == 0 && trace.model_a_actions[5] > 0 {
            self.model_a_preflop_allin_hands_by_tier[tier_idx] += 1;
        }
        if model_a_bb > 1e-12 {
            self.model_a_win_total_bb += model_a_bb;
            self.model_a_win_bucket_counts[result_bucket(model_a_bb.abs())] += 1;
            if trace.max_round == 0 && model_a_bb.abs() >= 20.0 {
                self.model_a_big_win_preflop_by_tier[tier_idx] += 1;
            }
            self.top_wins.push(trace);
            self.top_wins.sort_by(|a, b| {
                b.model_a_bb
                    .partial_cmp(&a.model_a_bb)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            self.top_wins.truncate(8);
        } else if model_a_bb < -1e-12 {
            self.model_a_loss_total_abs_bb += -model_a_bb;
            self.model_a_loss_bucket_counts[result_bucket(model_a_bb.abs())] += 1;
            if trace.max_round == 0 && model_a_bb.abs() >= 20.0 {
                self.model_a_big_loss_preflop_by_tier[tier_idx] += 1;
            }
            self.top_losses.push(trace);
            self.top_losses.sort_by(|a, b| {
                a.model_a_bb
                    .partial_cmp(&b.model_a_bb)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            self.top_losses.truncate(8);
        }
    }
}

#[derive(Clone, Debug, Default)]
struct HandTrace {
    hand_seed: u64,
    model_a_player: usize,
    model_a_bb: f64,
    model_b_bb: f64,
    max_round: usize,
    model_a_actions: [u32; DEBUG_ACTION_GROUPS],
    model_b_actions: [u32; DEBUG_ACTION_GROUPS],
    model_a_pressure: [u32; DEBUG_PRESSURE_BUCKETS],
    model_b_pressure: [u32; DEBUG_PRESSURE_BUCKETS],
    model_a_hole: Vec<String>,
    model_b_hole: Vec<String>,
    board: Vec<String>,
    model_a_preflop_tier: usize,
    model_b_preflop_tier: usize,
    model_a_decisions: Vec<String>,
    action_history: Vec<String>,
}

fn hand_traces_json(records: &[HandTrace]) -> String {
    let items = records
        .iter()
        .map(|record| {
            let history = record
                .action_history
                .iter()
                .map(|token| format!("\"{}\"", json_escape(token)))
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "{{\"hand_seed\":{},\"model_a_player\":{},\"model_a_bb\":{},\"model_b_bb\":{},\"max_round\":{},\"model_a_actions\":{:?},\"model_b_actions\":{:?},\"model_a_pressure\":{:?},\"model_b_pressure\":{:?},\"model_a_hole\":{:?},\"model_b_hole\":{:?},\"board\":{:?},\"model_a_preflop_tier\":{},\"model_b_preflop_tier\":{},\"model_a_decisions\":{:?},\"action_history\":[{}]}}",
                record.hand_seed,
                record.model_a_player,
                record.model_a_bb,
                record.model_b_bb,
                record.max_round,
                record.model_a_actions,
                record.model_b_actions,
                record.model_a_pressure,
                record.model_b_pressure,
                record.model_a_hole,
                record.model_b_hole,
                record.board,
                record.model_a_preflop_tier,
                record.model_b_preflop_tier,
                record.model_a_decisions,
                history
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("[{}]", items)
}

fn debug_log_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .unwrap_or_else(|| Path::new("."))
        .join(DEBUG_LOG_FILE)
}

fn json_escape(raw: &str) -> String {
    raw.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\r', "\\r")
        .replace('\n', "\\n")
}

fn append_debug_log(hypothesis_id: &str, location: &str, message: &str, data_json: &str) {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_millis())
        .unwrap_or(0);
    let line = format!(
        "{{\"sessionId\":\"{session}\",\"runId\":\"{run_id}\",\"hypothesisId\":\"{hypothesis}\",\"location\":\"{location}\",\"message\":\"{message}\",\"data\":{data},\"timestamp\":{timestamp}}}",
        session = DEBUG_SESSION_ID,
        run_id = DEBUG_RUN_ID,
        hypothesis = json_escape(hypothesis_id),
        location = json_escape(location),
        message = json_escape(message),
        data = data_json,
        timestamp = timestamp,
    );
    let _ = OpenOptions::new()
        .create(true)
        .append(true)
        .open(debug_log_path())
        .and_then(|mut handle| writeln!(handle, "{line}"));
}

fn round_index(round: BettingRound) -> usize {
    match round {
        BettingRound::Preflop => 0,
        BettingRound::Flop => 1,
        BettingRound::Turn => 2,
        BettingRound::River | BettingRound::Complete => 3,
    }
}

fn result_bucket(abs_bb: f64) -> usize {
    if abs_bb < 5.0 {
        0
    } else if abs_bb < 20.0 {
        1
    } else {
        2
    }
}

fn pressure_bucket(game: &game::NlheGame, player_idx: usize) -> usize {
    let legal = game.legal_actions(player_idx).unwrap_or_default();
    let to_call = legal
        .iter()
        .find_map(|action| match action {
            game::LegalAction::Call { amount } => Some(*amount),
            _ => None,
        })
        .unwrap_or(0);
    if to_call == 0 {
        return 0;
    }
    let pot: u32 = game
        .players
        .iter()
        .map(|player| player.total_contribution)
        .sum();
    let pressure = to_call as f64 / (pot.saturating_add(to_call).max(1) as f64);
    if pressure <= 0.10 {
        1
    } else if pressure <= 0.33 {
        2
    } else {
        3
    }
}

fn model_source_kind(source: &ModelSource) -> &'static str {
    match source {
        ModelSource::Onnx { .. } => "onnx",
        ModelSource::Tag => "tag",
        ModelSource::Random => "random",
    }
}

fn sibling_advantage_paths(model_a: &ModelSource) -> Option<[PathBuf; 2]> {
    let ModelSource::Onnx {
        path,
        policy_output,
    } = model_a
    else {
        return None;
    };
    if *policy_output != PolicyOutput::Strategy {
        return None;
    }
    let parent = path.parent()?;
    let p0 = parent.join("advantage_p0.onnx");
    let p1 = parent.join("advantage_p1.onnx");
    if p0.exists() && p1.exists() {
        Some([p0, p1])
    } else {
        None
    }
}

fn card_debug_string(card: game::Card) -> String {
    let rank = match card.rank {
        game::Rank::Two => "2",
        game::Rank::Three => "3",
        game::Rank::Four => "4",
        game::Rank::Five => "5",
        game::Rank::Six => "6",
        game::Rank::Seven => "7",
        game::Rank::Eight => "8",
        game::Rank::Nine => "9",
        game::Rank::Ten => "T",
        game::Rank::Jack => "J",
        game::Rank::Queen => "Q",
        game::Rank::King => "K",
        game::Rank::Ace => "A",
    };
    let suit = match card.suit {
        game::Suit::Clubs => "c",
        game::Suit::Diamonds => "d",
        game::Suit::Hearts => "h",
        game::Suit::Spades => "s",
    };
    format!("{rank}{suit}")
}

fn round_name(round: BettingRound) -> &'static str {
    match round {
        BettingRound::Preflop => "preflop",
        BettingRound::Flop => "flop",
        BettingRound::Turn => "turn",
        BettingRound::River => "river",
        BettingRound::Complete => "complete",
    }
}

fn preflop_tier_index_for_cards(hole: [game::Card; 2]) -> usize {
    let r0 = rank_value(hole[0].rank);
    let r1 = rank_value(hole[1].rank);
    let high = r0.max(r1);
    let low = r0.min(r1);
    let pair = r0 == r1;
    let suited = hole[0].suit == hole[1].suit;
    let gap = high.saturating_sub(low);

    if pair {
        return match high {
            12..=14 => 0,
            10..=11 => 1,
            5..=9 => 2,
            _ => 3,
        };
    }
    if high == 14 && low == 13 && suited {
        return 0;
    }
    if (high == 14 && low == 13) || (high == 14 && low == 12 && suited) {
        return 1;
    }
    let broadway_combo = high >= 11 && low >= 10;
    let strong_suited_connector = suited && gap <= 1 && high >= 9 && low >= 7;
    let strong_suited_ace = suited && high == 14 && low >= 10;
    if broadway_combo || strong_suited_connector || strong_suited_ace {
        return 2;
    }
    let weak_ace = high == 14;
    let small_suited_connector = suited && gap <= 2 && high >= 6;
    if weak_ace || small_suited_connector {
        return 3;
    }
    4
}

fn rank_value(rank: game::Rank) -> u8 {
    match rank {
        game::Rank::Two => 2,
        game::Rank::Three => 3,
        game::Rank::Four => 4,
        game::Rank::Five => 5,
        game::Rank::Six => 6,
        game::Rank::Seven => 7,
        game::Rank::Eight => 8,
        game::Rank::Nine => 9,
        game::Rank::Ten => 10,
        game::Rank::Jack => 11,
        game::Rank::Queen => 12,
        game::Rank::King => 13,
        game::Rank::Ace => 14,
    }
}

fn record_action(
    stats: &mut DebugStats,
    side_idx: usize,
    round: BettingRound,
    pressure_idx: usize,
    sort_group: u8,
) {
    let side = side_idx.min(1);
    let street = round_index(round);
    let pressure = pressure_idx.min(DEBUG_PRESSURE_BUCKETS - 1);
    let action_group = usize::from(sort_group).min(DEBUG_ACTION_GROUPS - 1);
    stats.action_counts[side][street][pressure][action_group] += 1;
}

#[derive(Clone, Debug)]
enum ModelSource {
    Onnx {
        path: PathBuf,
        policy_output: PolicyOutput,
    },
    Tag,
    Random,
}

impl ModelSource {
    fn display(&self) -> String {
        match self {
            Self::Onnx {
                path,
                policy_output,
            } => format!("{} (policy={})", path.display(), policy_output.as_str()),
            Self::Tag => "tag_rule_based".to_string(),
            Self::Random => "random_uniform".to_string(),
        }
    }
}

#[derive(Debug)]
struct Config {
    model_a: ModelSource,
    model_b: ModelSource,
    companion_advantage_a: bool,
    cluster_dir: PathBuf,
    hands: usize,
    deck_samples: usize,
    workers: usize,
    progress_every: usize,
    duplicate: bool,
    seed: u64,
    starting_stack: u32,
    small_blind: u32,
    big_blind: u32,
}

#[derive(Debug, Default)]
struct HeadToHeadReport {
    hands: usize,
    model_a_total_bb: f64,
    model_b_total_bb: f64,
    model_a_wins: u64,
    model_b_wins: u64,
    ties: u64,
}

enum PlayerPolicy {
    Onnx(OnnxPolicy),
    Tag(TagPolicy),
    Random(RandomPolicy),
}

struct RandomPolicy;

struct CompanionPolicies {
    by_player: [OnnxPolicy; 2],
}

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

fn build_companion_policies(
    worker_idx: usize,
    model_a: &ModelSource,
) -> AppResult<Option<CompanionPolicies>> {
    let Some(paths) = sibling_advantage_paths(model_a) else {
        return Ok(None);
    };
    let p0 = OnnxPolicy::from_file_with_output_mode(&paths[0], PolicyOutput::Advantage).map_err(
        |err| {
            format!(
                "worker {worker_idx} failed to load companion model_p0 {}: {err}",
                paths[0].display()
            )
        },
    )?;
    let p1 = OnnxPolicy::from_file_with_output_mode(&paths[1], PolicyOutput::Advantage).map_err(
        |err| {
            format!(
                "worker {worker_idx} failed to load companion model_p1 {}: {err}",
                paths[1].display()
            )
        },
    )?;
    Ok(Some(CompanionPolicies {
        by_player: [p0, p1],
    }))
}

impl PlayerPolicy {
    fn from_model_source(source: &ModelSource, worker_idx: usize, side: &str) -> AppResult<Self> {
        match source {
            ModelSource::Onnx {
                path,
                policy_output,
            } => {
                let policy = OnnxPolicy::from_file_with_output_mode(path, *policy_output).map_err(
                    |err| {
                        format!(
                            "worker {worker_idx} failed to load model_{side} {}: {err}",
                            path.display()
                        )
                    },
                )?;
                Ok(Self::Onnx(policy))
            }
            ModelSource::Tag => Ok(Self::Tag(TagPolicy::new())),
            ModelSource::Random => Ok(Self::Random(RandomPolicy::new())),
        }
    }

    fn get_strategy(
        &mut self,
        state: &cfr::nlhe_game::NlheState,
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

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
}

fn parse_optional_path_arg(args: &[String], name: &str) -> Option<PathBuf> {
    flag_value(args, name).map(PathBuf::from)
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

fn parse_config(args: &[String]) -> AppResult<Config> {
    let tag_a = has_flag(args, "--tag-a");
    let random_a = has_flag(args, "--random-a");
    let tag_b = has_flag(args, "--tag-b");
    let random_b = has_flag(args, "--random-b");
    let companion_advantage_a = has_flag(args, "--companion-advantage-a");
    let model_a_path = parse_optional_path_arg(args, "--model-a");
    let model_b_path = parse_optional_path_arg(args, "--model-b");
    let policy_a = parse_policy_output_arg(args, "--policy-a", PolicyOutput::Advantage)?;
    let policy_b = parse_policy_output_arg(args, "--policy-b", PolicyOutput::Advantage)?;
    let cluster_dir = parse_path_arg(args, "--cluster-dir", DEFAULT_CLUSTER_DIR);
    let hands = parse_usize_arg(args, "--hands", DEFAULT_HANDS)?;
    let deck_samples = parse_usize_arg(args, "--deck-samples", DEFAULT_DECK_SAMPLES)?;
    let workers = parse_usize_arg(args, "--workers", DEFAULT_WORKERS)?;
    let progress_every = parse_usize_arg(args, "--progress-every", DEFAULT_PROGRESS_EVERY)?;
    let duplicate = !has_flag(args, "--no-duplicate");
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
    if big_blind == 0 {
        return Err("--big-blind must be greater than 0".to_string());
    }
    if small_blind == 0 {
        return Err("--small-blind must be greater than 0".to_string());
    }
    if starting_stack == 0 {
        return Err("--starting-stack must be greater than 0".to_string());
    }
    if tag_a && random_a {
        return Err("cannot set both --tag-a and --random-a".to_string());
    }
    if tag_b && random_b {
        return Err("cannot set both --tag-b and --random-b".to_string());
    }

    let model_a = if tag_a {
        ModelSource::Tag
    } else if random_a {
        ModelSource::Random
    } else {
        let path = model_a_path.ok_or_else(|| {
            "missing required flag: --model-a (or use --tag-a/--random-a)".to_string()
        })?;
        ModelSource::Onnx {
            path,
            policy_output: policy_a,
        }
    };
    let model_b = if tag_b {
        ModelSource::Tag
    } else if random_b {
        ModelSource::Random
    } else {
        let path = model_b_path.ok_or_else(|| {
            "missing required flag: --model-b (or use --tag-b/--random-b)".to_string()
        })?;
        ModelSource::Onnx {
            path,
            policy_output: policy_b,
        }
    };

    Ok(Config {
        model_a,
        model_b,
        companion_advantage_a,
        cluster_dir,
        hands,
        deck_samples,
        workers,
        progress_every,
        duplicate,
        seed,
        starting_stack,
        small_blind,
        big_blind,
    })
}

fn print_help() {
    println!(
        "head_to_head options:\n\
         --model-a <path> (required unless --tag-a/--random-a)\n\
         --policy-a <advantage|strategy> (default advantage; ignored with --tag-a/--random-a)\n\
         --companion-advantage-a (when model_a is strategy, use sibling advantage_p0/advantage_p1 for actual play)\n\
         --tag-a (use rule-based TAG policy for model A)\n\
         --random-a (use uniform random policy for model A)\n\
         --model-b <path> (required unless --tag-b/--random-b)\n\
         --policy-b <advantage|strategy> (default advantage; ignored with --tag-b/--random-b)\n\
         --tag-b (use rule-based TAG policy for model B)\n\
         --random-b (use uniform random policy for model B)\n\
         --cluster-dir <path> (default {DEFAULT_CLUSTER_DIR})\n\
         --hands <usize> (default {DEFAULT_HANDS})\n\
         --deck-samples <usize> (default {DEFAULT_DECK_SAMPLES})\n\
         --workers <usize> (default {DEFAULT_WORKERS}; 0 means auto)\n\
         --progress-every <usize> (default {DEFAULT_PROGRESS_EVERY}; 0 disables periodic progress)\n\
         --no-duplicate (disable paired duplicate-seat hand evaluation)\n\
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
                "[h2h] progress: {done_clamped}/{total} ({pct:.1}%) elapsed={} eta~{} speed={hands_per_sec:.1} hands/s",
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

fn simulate_hand(
    game: &NlheGameModel,
    model_a_player: usize,
    policy_a: &mut PlayerPolicy,
    policy_b: &mut PlayerPolicy,
    seed: u64,
    debug_stats: &mut DebugStats,
    companion_policies: &mut Option<CompanionPolicies>,
    use_companion_for_model_a: bool,
) -> AppResult<(f64, f64)> {
    let model_b_player = if model_a_player == 0 { 1 } else { 0 };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state: NlheState = game.root_state();
    let mut trace = HandTrace {
        hand_seed: seed,
        model_a_player,
        ..HandTrace::default()
    };

    loop {
        match game.node_kind(&state) {
            NodeKind::Terminal => {
                let a = game.terminal_utility(&state, model_a_player);
                let b = game.terminal_utility(&state, model_b_player);
                trace.model_a_bb = a;
                trace.model_b_bb = b;
                trace.action_history = state.action_history.clone();
                if let Some(game_state) = state.game.as_ref() {
                    let model_a_hole_cards = game_state.players[model_a_player].hole_cards;
                    let model_b_hole_cards = game_state.players[model_b_player].hole_cards;
                    trace.model_a_hole = model_a_hole_cards
                        .iter()
                        .copied()
                        .map(card_debug_string)
                        .collect();
                    trace.model_b_hole = model_b_hole_cards
                        .iter()
                        .copied()
                        .map(card_debug_string)
                        .collect();
                    trace.board = game_state
                        .board
                        .iter()
                        .copied()
                        .map(card_debug_string)
                        .collect();
                    trace.model_a_preflop_tier = preflop_tier_index_for_cards(model_a_hole_cards);
                    trace.model_b_preflop_tier = preflop_tier_index_for_cards(model_b_hole_cards);
                }
                debug_stats.record_result(trace);
                return Ok((a, b));
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
                    let a = game.terminal_utility(&state, model_a_player);
                    let b = game.terminal_utility(&state, model_b_player);
                    return Ok((a, b));
                }

                let strategy = if player_idx == model_a_player {
                    if use_companion_for_model_a {
                        if let Some(companion) = companion_policies.as_mut() {
                            let features =
                                encode_nlhe_state(&state, player_idx).ok_or_else(|| {
                                    "expected resolved NLHE state for companion encoding"
                                        .to_string()
                                })?;
                            let (action_slots, action_mask) = action_slots_and_mask(&action_space)?;
                            companion.by_player[player_idx]
                                .get_strategy(&features, &action_slots, &action_mask)
                                .map_err(|err| {
                                    format!(
                                        "failed to query companion strategy for player {player_idx}: {err}"
                                    )
                                })?
                        } else {
                            return Err(
                                "companion advantage policy requested but unavailable".to_string()
                            );
                        }
                    } else {
                        policy_a
                            .get_strategy(&state, player_idx, &action_space)
                            .map_err(|err| {
                                format!("failed to query strategy for player {player_idx}: {err}")
                            })?
                    }
                } else {
                    policy_b
                        .get_strategy(&state, player_idx, &action_space)
                        .map_err(|err| {
                            format!("failed to query strategy for player {player_idx}: {err}")
                        })?
                };

                if strategy.len() != action_count {
                    return Err(format!(
                        "strategy/action count mismatch: strategy={}, actions={action_count}",
                        strategy.len()
                    ));
                }

                let pressure_idx = pressure_bucket(game_state, player_idx);
                if player_idx == model_a_player {
                    let probs = strategy
                        .iter()
                        .enumerate()
                        .map(|(idx, prob)| {
                            let token = action_space
                                .get(idx)
                                .map(|action| action.action_token.as_str())
                                .unwrap_or("?");
                            format!("{token}:{prob:.3}")
                        })
                        .collect::<Vec<_>>()
                        .join(",");
                    let companion_text = if let Some(companion) = companion_policies.as_mut() {
                        let features = encode_nlhe_state(&state, player_idx).ok_or_else(|| {
                            "expected resolved NLHE state for companion encoding".to_string()
                        })?;
                        let (action_slots, action_mask) = action_slots_and_mask(&action_space)?;
                        let adv_probs = companion.by_player[player_idx]
                            .get_strategy(&features, &action_slots, &action_mask)
                            .map_err(|err| {
                                format!(
                                    "failed to query companion advantage strategy for player {player_idx}: {err}"
                                )
                            })?;
                        let adv_text = adv_probs
                            .iter()
                            .enumerate()
                            .map(|(idx, prob)| {
                                let token = action_space
                                    .get(idx)
                                    .map(|action| action.action_token.as_str())
                                    .unwrap_or("?");
                                format!("{token}:{prob:.3}")
                            })
                            .collect::<Vec<_>>()
                            .join(",");
                        format!("|adv=[{}]", adv_text)
                    } else {
                        String::new()
                    };
                    trace.model_a_decisions.push(format!(
                        "{}|pressure={}|choices=[{}]{}",
                        round_name(game_state.round),
                        pressure_idx,
                        probs,
                        companion_text
                    ));
                }
                let action_idx = sample_from_probs(&strategy, &mut rng);
                let side_idx = if player_idx == model_a_player { 0 } else { 1 };
                let sort_group = action_space
                    .get(action_idx)
                    .map(|action| action.sort_group)
                    .unwrap_or(0);
                // #region agent log
                record_action(
                    debug_stats,
                    side_idx,
                    game_state.round,
                    pressure_idx,
                    sort_group,
                );
                // #endregion
                let action_group = usize::from(sort_group).min(DEBUG_ACTION_GROUPS - 1);
                if side_idx == 0 {
                    trace.model_a_actions[action_group] += 1;
                    trace.model_a_pressure[pressure_idx.min(DEBUG_PRESSURE_BUCKETS - 1)] += 1;
                } else {
                    trace.model_b_actions[action_group] += 1;
                    trace.model_b_pressure[pressure_idx.min(DEBUG_PRESSURE_BUCKETS - 1)] += 1;
                }
                trace.max_round = trace.max_round.max(round_index(game_state.round));
                state = game.next_state(&state, action_idx);
            }
        }
    }
}

fn evaluate_head_to_head(
    config: &Config,
    game: &NlheGameModel,
    pool: &rayon::ThreadPool,
    worker_count: usize,
) -> AppResult<(HeadToHeadReport, DebugStats)> {
    let hands: Vec<usize> = (0..config.hands).collect();
    let chunk_size = hand_chunk_size(hands.len(), worker_count);
    let started = Instant::now();
    let done = AtomicUsize::new(0);
    let next_report = AtomicUsize::new(config.progress_every.max(1));

    println!(
        "[h2h] start: hands={} workers={} chunk_size={}",
        config.hands, worker_count, chunk_size
    );

    let partials = pool.install(|| {
        hands
            .par_chunks(chunk_size)
            .enumerate()
            .map(
                |(worker_idx, chunk)| -> AppResult<(f64, f64, u64, u64, u64, DebugStats)> {
                    let mut policy_a =
                        PlayerPolicy::from_model_source(&config.model_a, worker_idx, "a")?;
                    let mut policy_b =
                        PlayerPolicy::from_model_source(&config.model_b, worker_idx, "b")?;
                    let mut companion_policies =
                        build_companion_policies(worker_idx, &config.model_a)?;

                    let mut total_a = 0.0f64;
                    let mut total_b = 0.0f64;
                    let mut wins_a = 0u64;
                    let mut wins_b = 0u64;
                    let mut ties = 0u64;
                    let mut local_debug = DebugStats::default();

                    for hand_idx in chunk {
                        let (model_a_player, hand_seed) = if config.duplicate {
                            // Pair adjacent hands with the same chance seed while swapping seats.
                            // This lowers variance for checkpoint-vs-checkpoint comparisons.
                            let model_a_player = if hand_idx % 2 == 0 { 0 } else { 1 };
                            let pair_idx = hand_idx / 2;
                            let hand_seed = config.seed
                                ^ ((pair_idx as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03));
                            (model_a_player, hand_seed)
                        } else {
                            let model_a_player = if hand_idx % 2 == 0 { 0 } else { 1 };
                            let hand_seed = config.seed
                                ^ ((*hand_idx as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03))
                                ^ ((model_a_player as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
                            (model_a_player, hand_seed)
                        };

                        let (a, b) = simulate_hand(
                            game,
                            model_a_player,
                            &mut policy_a,
                            &mut policy_b,
                            hand_seed,
                            &mut local_debug,
                            &mut companion_policies,
                            config.companion_advantage_a,
                        )?;
                        total_a += a;
                        total_b += b;

                        if a > 1e-12 {
                            wins_a += 1;
                        } else if a < -1e-12 {
                            wins_b += 1;
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

                    Ok((total_a, total_b, wins_a, wins_b, ties, local_debug))
                },
            )
            .collect::<Vec<_>>()
    });

    let mut report = HeadToHeadReport {
        hands: config.hands,
        ..HeadToHeadReport::default()
    };
    let mut debug_stats = DebugStats::default();
    for partial in partials {
        let (total_a, total_b, wins_a, wins_b, ties, partial_debug) = partial?;
        report.model_a_total_bb += total_a;
        report.model_b_total_bb += total_b;
        report.model_a_wins += wins_a;
        report.model_b_wins += wins_b;
        report.ties += ties;
        debug_stats.merge(partial_debug);
    }

    println!(
        "[h2h] done: {} hands in {}",
        report.hands,
        format_duration(started.elapsed().as_secs_f64())
    );
    Ok((report, debug_stats))
}

fn run_main() -> AppResult<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_help();
        return Ok(());
    }
    let config = parse_config(&args)?;
    let worker_count = resolve_worker_count(config.workers, config.hands);
    println!("[h2h] model_a={}", config.model_a.display());
    println!("[h2h] model_b={}", config.model_b.display());
    println!(
        "[h2h] game: 2-player {:.1}bb (sb={} bb={} stack={})",
        config.starting_stack as f64 / config.big_blind as f64,
        config.small_blind,
        config.big_blind,
        config.starting_stack
    );
    println!(
        "[h2h] hands={} deck_samples={} workers={} seed={} progress_every={} duplicate={}",
        config.hands,
        config.deck_samples,
        worker_count,
        config.seed,
        config.progress_every,
        config.duplicate
    );

    let deck_seeds = build_deck_seeds(config.deck_samples, config.seed ^ 0xA24B_AED4_963E_E407);
    let game = NlheGameModel::new(
        NlheConfig {
            num_players: 2,
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
    let (report, debug_stats) = evaluate_head_to_head(&config, &game, &pool, worker_count)?;
    let elapsed = started.elapsed().as_secs_f64();
    let model_a_bb_per_hand = report.model_a_total_bb / report.hands as f64;
    let model_b_bb_per_hand = report.model_b_total_bb / report.hands as f64;
    let model_a_bb_per_100 = model_a_bb_per_hand * 100.0;
    let model_b_bb_per_100 = model_b_bb_per_hand * 100.0;

    println!("[h2h] results:");
    println!(
        "[h2h]   model_a: {model_a_bb_per_hand:+.6} bb/hand ({model_a_bb_per_100:+.3} bb/100)"
    );
    println!(
        "[h2h]   model_b: {model_b_bb_per_hand:+.6} bb/hand ({model_b_bb_per_100:+.3} bb/100)"
    );
    println!(
        "[h2h]   outcomes: model_a_wins={} model_b_wins={} ties={}",
        report.model_a_wins, report.model_b_wins, report.ties
    );
    println!(
        "[h2h]   zero_sum_check={:.9} bb/hand",
        model_a_bb_per_hand + model_b_bb_per_hand
    );
    println!("[h2h]   hands={} elapsed={elapsed:.1}s", report.hands);

    // #region agent log
    append_debug_log(
        "H2",
        "head_to_head.rs:run_main:result_buckets",
        "model_a_result_bucket_summary",
        &format!(
            "{{\"model_b_kind\":\"{}\",\"model_a_wins\":{},\"model_a_losses\":{},\"ties\":{},\"model_a_win_total_bb\":{},\"model_a_loss_total_abs_bb\":{},\"model_a_win_bucket_counts\":{:?},\"model_a_loss_bucket_counts\":{:?}}}",
            json_escape(model_source_kind(&config.model_b)),
            report.model_a_wins,
            report.model_b_wins,
            report.ties,
            debug_stats.model_a_win_total_bb,
            debug_stats.model_a_loss_total_abs_bb,
            debug_stats.model_a_win_bucket_counts,
            debug_stats.model_a_loss_bucket_counts
        ),
    );
    // #endregion

    // #region agent log
    append_debug_log(
        "H4",
        "head_to_head.rs:run_main:action_profile",
        "action_profile_by_side_round_pressure",
        &format!(
            "{{\"model_b_kind\":\"{}\",\"action_counts\":{:?}}}",
            json_escape(model_source_kind(&config.model_b)),
            debug_stats.action_counts
        ),
    );
    // #endregion

    // #region agent log
    append_debug_log(
        "H7",
        "head_to_head.rs:run_main:preflop_tier_summary",
        "preflop_aggression_by_tier",
        &format!(
            "{{\"model_b_kind\":\"{}\",\"model_a_preflop_aggro_hands_by_tier\":{:?},\"model_a_preflop_allin_hands_by_tier\":{:?},\"model_a_big_loss_preflop_by_tier\":{:?},\"model_a_big_win_preflop_by_tier\":{:?}}}",
            json_escape(model_source_kind(&config.model_b)),
            debug_stats.model_a_preflop_aggro_hands_by_tier,
            debug_stats.model_a_preflop_allin_hands_by_tier,
            debug_stats.model_a_big_loss_preflop_by_tier,
            debug_stats.model_a_big_win_preflop_by_tier
        ),
    );
    // #endregion

    // #region agent log
    append_debug_log(
        "H5",
        "head_to_head.rs:run_main:top_losses",
        "top_model_a_losses",
        &format!(
            "{{\"model_b_kind\":\"{}\",\"records\":{}}}",
            json_escape(model_source_kind(&config.model_b)),
            hand_traces_json(&debug_stats.top_losses)
        ),
    );
    // #endregion

    // #region agent log
    append_debug_log(
        "H6",
        "head_to_head.rs:run_main:top_wins",
        "top_model_a_wins",
        &format!(
            "{{\"model_b_kind\":\"{}\",\"records\":{}}}",
            json_escape(model_source_kind(&config.model_b)),
            hand_traces_json(&debug_stats.top_wins)
        ),
    );
    // #endregion

    Ok(())
}

fn main() {
    let builder = std::thread::Builder::new()
        .name("deep-cfr-head-to-head-main".to_string())
        .stack_size(64 * 1024 * 1024);
    let handle = builder
        .spawn(run_main)
        .unwrap_or_else(|err| panic!("failed to spawn head_to_head thread: {err}"));

    match handle.join() {
        Ok(Ok(())) => {}
        Ok(Err(err)) => {
            eprintln!("[h2h] error: {err}");
            std::process::exit(1);
        }
        Err(_) => panic!("head_to_head thread panicked"),
    }
}
