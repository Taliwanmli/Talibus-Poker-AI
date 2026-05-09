use crate::encoding::encode_nlhe_state;
use crate::onnx_policy::OnnxPolicyError;
use crate::sample::{AdvantageSample, StrategySample, MAX_ACTIONS};
use cfr::external_sampling::{GameModel, NodeKind};
use cfr::nlhe_game::{enumerate_action_space, IndexedAction, NlheState};
use game::{Card, LegalAction, Rank};
use rand::prelude::{Rng, SeedableRng, StdRng};
use std::sync::{Mutex, OnceLock};

#[derive(Debug)]
pub enum TraverseError {
    Policy(OnnxPolicyError),
    MissingGameState,
    InvalidActionCount(usize),
    InvalidPolicySlot(usize),
    DuplicatePolicySlot(usize),
    InvalidPlayerIndex(usize),
}

impl std::fmt::Display for TraverseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Policy(err) => write!(f, "{err}"),
            Self::MissingGameState => write!(f, "expected resolved NLHE game state for encoding"),
            Self::InvalidActionCount(count) => write!(
                f,
                "action count {count} exceeds deep CFR max action slots {MAX_ACTIONS}"
            ),
            Self::InvalidPolicySlot(slot) => {
                write!(f, "action policy slot {slot} is out of range")
            }
            Self::DuplicatePolicySlot(slot) => {
                write!(f, "action policy slot {slot} appears multiple times")
            }
            Self::InvalidPlayerIndex(player_idx) => {
                write!(f, "policy vector missing player index {player_idx}")
            }
        }
    }
}

impl std::error::Error for TraverseError {}

impl From<OnnxPolicyError> for TraverseError {
    fn from(value: OnnxPolicyError) -> Self {
        Self::Policy(value)
    }
}

pub type Result<T> = std::result::Result<T, TraverseError>;

pub trait PolicyProvider {
    fn get_strategy(
        &mut self,
        features: &[f32],
        action_slots: &[usize],
        action_mask: &[f32; MAX_ACTIONS],
    ) -> std::result::Result<Vec<f64>, OnnxPolicyError>;
}

const DEBUG_PRE_TIER_COUNT: usize = 5;
const DEBUG_PRESSURE_BUCKETS: usize = 4;
const DEBUG_ACTION_GROUPS: usize = 6;

#[derive(Clone, Debug)]
pub struct TraverserPreflopDebugSnapshot {
    pub total_nodes: u64,
    pub tier_counts: [u64; DEBUG_PRE_TIER_COUNT],
    pub best_group_counts:
        [[[u64; DEBUG_ACTION_GROUPS]; DEBUG_PRESSURE_BUCKETS]; DEBUG_PRE_TIER_COUNT],
}

impl Default for TraverserPreflopDebugSnapshot {
    fn default() -> Self {
        Self {
            total_nodes: 0,
            tier_counts: [0; DEBUG_PRE_TIER_COUNT],
            best_group_counts: [[[0; DEBUG_ACTION_GROUPS]; DEBUG_PRESSURE_BUCKETS];
                DEBUG_PRE_TIER_COUNT],
        }
    }
}

static TRAVERSER_PREFLOP_DEBUG: OnceLock<Mutex<TraverserPreflopDebugSnapshot>> = OnceLock::new();
#[derive(Clone, Debug)]
pub struct OpponentPreflopDebugSnapshot {
    pub total_nodes: u64,
    pub tier_counts: [u64; DEBUG_PRE_TIER_COUNT],
    pub action_mass: [[[f64; DEBUG_ACTION_GROUPS]; DEBUG_PRESSURE_BUCKETS]; DEBUG_PRE_TIER_COUNT],
}

impl Default for OpponentPreflopDebugSnapshot {
    fn default() -> Self {
        Self {
            total_nodes: 0,
            tier_counts: [0; DEBUG_PRE_TIER_COUNT],
            action_mass: [[[0.0; DEBUG_ACTION_GROUPS]; DEBUG_PRESSURE_BUCKETS];
                DEBUG_PRE_TIER_COUNT],
        }
    }
}

static OPPONENT_PREFLOP_DEBUG: OnceLock<Mutex<OpponentPreflopDebugSnapshot>> = OnceLock::new();

fn traverser_preflop_debug() -> &'static Mutex<TraverserPreflopDebugSnapshot> {
    TRAVERSER_PREFLOP_DEBUG.get_or_init(|| Mutex::new(TraverserPreflopDebugSnapshot::default()))
}

fn opponent_preflop_debug() -> &'static Mutex<OpponentPreflopDebugSnapshot> {
    OPPONENT_PREFLOP_DEBUG.get_or_init(|| Mutex::new(OpponentPreflopDebugSnapshot::default()))
}

pub fn take_traverser_preflop_debug_snapshot() -> Option<TraverserPreflopDebugSnapshot> {
    let mut guard = traverser_preflop_debug().lock().ok()?;
    if guard.total_nodes == 0 {
        return None;
    }
    let snapshot = guard.clone();
    *guard = TraverserPreflopDebugSnapshot::default();
    Some(snapshot)
}

pub fn take_opponent_preflop_debug_snapshot() -> Option<OpponentPreflopDebugSnapshot> {
    let mut guard = opponent_preflop_debug().lock().ok()?;
    if guard.total_nodes == 0 {
        return None;
    }
    let snapshot = guard.clone();
    *guard = OpponentPreflopDebugSnapshot::default();
    Some(snapshot)
}

#[derive(Debug, Clone, Default)]
pub struct TraversalStats {
    pub traverser_nodes: u64,
    pub opponent_nodes: u64,
    pub strategy_samples: u64,
    pub chance_nodes: u64,
    pub terminal_nodes: u64,
    pub opponent_actions_sampled: u64,
}

impl TraversalStats {
    pub fn merge_from(&mut self, other: &Self) {
        self.traverser_nodes += other.traverser_nodes;
        self.opponent_nodes += other.opponent_nodes;
        self.strategy_samples += other.strategy_samples;
        self.chance_nodes += other.chance_nodes;
        self.terminal_nodes += other.terminal_nodes;
        self.opponent_actions_sampled += other.opponent_actions_sampled;
    }
}

pub fn run_traversal_batch<G: GameModel<State = NlheState>, P: PolicyProvider>(
    game: &G,
    policies: &mut [P],
    traverser: usize,
    traversals: usize,
    seed: u64,
    iteration: u32,
) -> Result<(Vec<AdvantageSample>, Vec<StrategySample>, TraversalStats)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut advantage_samples = Vec::new();
    let mut strategy_samples = Vec::new();
    let mut stats = TraversalStats::default();
    for _ in 0..traversals {
        let root = game.root_state();
        let _ = traverse_deep_cfr(
            game,
            root,
            traverser,
            policies,
            &mut advantage_samples,
            &mut strategy_samples,
            &mut stats,
            &mut rng,
            iteration,
        )?;
    }
    Ok((advantage_samples, strategy_samples, stats))
}

pub fn traverse_deep_cfr<G: GameModel<State = NlheState>, P: PolicyProvider>(
    game: &G,
    state: G::State,
    traverser: usize,
    policies: &mut [P],
    advantage_samples: &mut Vec<AdvantageSample>,
    strategy_samples: &mut Vec<StrategySample>,
    stats: &mut TraversalStats,
    rng: &mut StdRng,
    iteration: u32,
) -> Result<f64> {
    match game.node_kind(&state) {
        NodeKind::Terminal => {
            stats.terminal_nodes += 1;
            Ok(game.terminal_utility(&state, traverser))
        }
        NodeKind::Chance => {
            stats.chance_nodes += 1;
            let probs = game.chance_probabilities(&state);
            let sampled = sample_from_probs(&probs, rng);
            let next = game.next_state(&state, sampled);
            traverse_deep_cfr(
                game,
                next,
                traverser,
                policies,
                advantage_samples,
                strategy_samples,
                stats,
                rng,
                iteration,
            )
        }
        NodeKind::Player(player_idx) => {
            let game_state = state.game.as_ref().ok_or(TraverseError::MissingGameState)?;
            let action_space = enumerate_action_space(game_state, player_idx);
            let action_count = action_space.len();
            if action_count == 0 {
                stats.terminal_nodes += 1;
                return Ok(game.terminal_utility(&state, traverser));
            }
            if action_count > MAX_ACTIONS {
                return Err(TraverseError::InvalidActionCount(action_count));
            }
            let (action_slots, action_mask_f32, action_mask_u8) =
                action_slots_and_masks(&action_space)?;

            let features =
                encode_nlhe_state(&state, player_idx).ok_or(TraverseError::MissingGameState)?;
            let strategy = {
                let policy = policies
                    .get_mut(player_idx)
                    .ok_or(TraverseError::InvalidPlayerIndex(player_idx))?;
                policy.get_strategy(&features, &action_slots, &action_mask_f32)?
            };

            if player_idx == traverser {
                stats.traverser_nodes += 1;
                let debug_meta = if game_state.round != game::BettingRound::Preflop {
                    None
                } else {
                    Some((
                        preflop_tier_index(game_state.players[player_idx].hole_cards),
                        pressure_bucket(game_state, player_idx),
                        action_space
                            .iter()
                            .map(|action| {
                                usize::from(action.sort_group).min(DEBUG_ACTION_GROUPS - 1)
                            })
                            .collect::<Vec<_>>(),
                    ))
                };
                let mut action_values = vec![0.0f64; action_count];
                let mut node_value = 0.0f64;
                for action_idx in 0..action_count {
                    let next = game.next_state(&state, action_idx);
                    action_values[action_idx] = traverse_deep_cfr(
                        game,
                        next,
                        traverser,
                        policies,
                        advantage_samples,
                        strategy_samples,
                        stats,
                        rng,
                        iteration,
                    )?;
                    node_value += strategy[action_idx] * action_values[action_idx];
                }

                let mut advantages = [0.0f32; MAX_ACTIONS];
                for action_idx in 0..action_count {
                    let slot = action_slots[action_idx];
                    advantages[slot] = (action_values[action_idx] - node_value) as f32;
                }
                // #region agent log
                if let Some((tier_idx, pressure_idx, sort_groups)) = debug_meta {
                    if let Some((best_idx, _)) =
                        action_values.iter().enumerate().max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                    {
                        if let Ok(mut guard) = traverser_preflop_debug().lock() {
                            guard.total_nodes += 1;
                            guard.tier_counts[tier_idx] += 1;
                            let best_group = *sort_groups.get(best_idx).unwrap_or(&0);
                            guard.best_group_counts[tier_idx][pressure_idx][best_group] += 1;
                        }
                    }
                }
                // #endregion
                advantage_samples.push(AdvantageSample::new(
                    features,
                    advantages,
                    action_mask_u8,
                    iteration,
                ));
                Ok(node_value)
            } else {
                stats.opponent_nodes += 1;
                // #region agent log
                if game_state.round == game::BettingRound::Preflop {
                    let tier_idx = preflop_tier_index(game_state.players[player_idx].hole_cards);
                    let pressure_idx = pressure_bucket(game_state, player_idx);
                    if let Ok(mut guard) = opponent_preflop_debug().lock() {
                        guard.total_nodes += 1;
                        guard.tier_counts[tier_idx] += 1;
                        for (idx, prob) in strategy.iter().enumerate().take(action_count) {
                            let action_group = action_space
                                .get(idx)
                                .map(|action| {
                                    usize::from(action.sort_group).min(DEBUG_ACTION_GROUPS - 1)
                                })
                                .unwrap_or(0);
                            guard.action_mass[tier_idx][pressure_idx][action_group] += *prob;
                        }
                    }
                }
                // #endregion
                let mut strategy_target = [0.0f32; MAX_ACTIONS];
                for (idx, prob) in strategy.iter().enumerate().take(action_count) {
                    let slot = action_slots[idx];
                    strategy_target[slot] = *prob as f32;
                }
                strategy_samples.push(StrategySample::new(
                    features,
                    strategy_target,
                    action_mask_u8,
                    iteration,
                ));
                stats.strategy_samples += 1;
                let sampled_action = sample_from_probs(&strategy, rng);
                stats.opponent_actions_sampled += 1;
                let next = game.next_state(&state, sampled_action);
                traverse_deep_cfr(
                    game,
                    next,
                    traverser,
                    policies,
                    advantage_samples,
                    strategy_samples,
                    stats,
                    rng,
                    iteration,
                )
            }
        }
    }
}

fn action_slots_and_masks(
    action_space: &[IndexedAction],
) -> Result<(Vec<usize>, [f32; MAX_ACTIONS], [u8; MAX_ACTIONS])> {
    let mut slots = Vec::with_capacity(action_space.len());
    let mut mask_f32 = [0.0f32; MAX_ACTIONS];
    let mut mask_u8 = [0u8; MAX_ACTIONS];
    for action in action_space {
        let slot = usize::from(action.policy_slot);
        if slot >= MAX_ACTIONS {
            return Err(TraverseError::InvalidPolicySlot(slot));
        }
        if mask_u8[slot] != 0 {
            return Err(TraverseError::DuplicatePolicySlot(slot));
        }
        mask_f32[slot] = 1.0;
        mask_u8[slot] = 1;
        slots.push(slot);
    }
    Ok((slots, mask_f32, mask_u8))
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

fn preflop_tier_index(hole: [Card; 2]) -> usize {
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

fn rank_value(rank: Rank) -> u8 {
    match rank {
        Rank::Two => 2,
        Rank::Three => 3,
        Rank::Four => 4,
        Rank::Five => 5,
        Rank::Six => 6,
        Rank::Seven => 7,
        Rank::Eight => 8,
        Rank::Nine => 9,
        Rank::Ten => 10,
        Rank::Jack => 11,
        Rank::Queen => 12,
        Rank::King => 13,
        Rank::Ace => 14,
    }
}

fn pressure_bucket(game: &game::NlheGame, player_idx: usize) -> usize {
    let legal = game.legal_actions(player_idx).unwrap_or_default();
    let to_call = legal
        .iter()
        .find_map(|action| match action {
            LegalAction::Call { amount } => Some(*amount),
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

#[cfg(test)]
mod tests {
    use super::action_slots_and_masks;
    use cfr::nlhe_game::IndexedAction;
    use game::Action;

    #[test]
    fn action_slot_masks_support_non_prefix_semantics() {
        let actions = vec![
            IndexedAction {
                engine_action: Action::Check,
                action_token: "x".to_string(),
                sort_group: 1,
                sort_amount: 0,
                policy_slot: 1,
            },
            IndexedAction {
                engine_action: Action::RaiseTo(120),
                action_token: "r:3bet_2.50x".to_string(),
                sort_group: 4,
                sort_amount: 120,
                policy_slot: 7,
            },
            IndexedAction {
                engine_action: Action::AllIn,
                action_token: "ai".to_string(),
                sort_group: 5,
                sort_amount: 200,
                policy_slot: 9,
            },
        ];
        let (slots, mask_f32, mask_u8) =
            action_slots_and_masks(&actions).expect("valid non-prefix slots");
        assert_eq!(slots, vec![1, 7, 9]);
        assert_eq!(mask_u8[0], 0);
        assert_eq!(mask_u8[1], 1);
        assert_eq!(mask_u8[7], 1);
        assert_eq!(mask_u8[9], 1);
        assert!((mask_f32[1] - 1.0).abs() < f32::EPSILON);
        assert!((mask_f32[7] - 1.0).abs() < f32::EPSILON);
        assert!((mask_f32[9] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn action_slot_masks_reject_duplicate_slots() {
        let actions = vec![
            IndexedAction {
                engine_action: Action::Check,
                action_token: "x".to_string(),
                sort_group: 1,
                sort_amount: 0,
                policy_slot: 1,
            },
            IndexedAction {
                engine_action: Action::Call,
                action_token: "c".to_string(),
                sort_group: 2,
                sort_amount: 10,
                policy_slot: 1,
            },
        ];
        let err = action_slots_and_masks(&actions).expect_err("duplicate slots should fail");
        assert!(
            matches!(err, super::TraverseError::DuplicatePolicySlot(1)),
            "expected duplicate slot error, got {err}"
        );
    }
}
