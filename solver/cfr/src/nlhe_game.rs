use crate::external_sampling::{GameModel, NodeKind};
use abstraction::action_abstraction::{
    map_real_bet_to_abstract, preflop_open_raise_grid, preflop_reraise_multiplier_grid,
    street_bet_grid, street_raise_multiplier_grid, AbstractAction, Street as AbstractStreet,
};
use abstraction::clustering::{
    buckets_for_street, canonical_preflop_index, canonicalize_preflop_hand, emd_distance,
    load_clustering_result, BucketStreet, ClusteringResult,
};
use abstraction::equity::{
    compute_board_texture_equity_multiway_with_seed, compute_equity_multiway_with_seed,
    equity_to_histogram, EQUITY_DIMS,
};
use dashmap::DashMap;
use game::{Action, BettingRound, LegalAction, NlheConfig, NlheGame, PlayerState, Rank};
use rs_poker::core::{Card as RsCard, Suit as RsSuit, Value as RsValue};
use std::cmp::Ordering;
use std::collections::{hash_map::DefaultHasher, HashSet};
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

const RUNTIME_EQUITY_SAMPLES: usize = 200;
const RUNTIME_BOARD_TEXTURE_SAMPLES: usize = 200;
const BOARD_TEXTURE_BUCKETS: u16 = 16;
pub const POLICY_MAX_ACTIONS: usize = 13;
const POLICY_BET_SLOT_START: u8 = 3;
const POLICY_BET_SLOT_COUNT: usize = 4;
const POLICY_RAISE_SLOT_START: u8 = 7;
const POLICY_RAISE_SLOT_COUNT: usize = 5;
const POLICY_ALL_IN_SLOT: u8 = 12;

#[derive(Clone)]
pub struct NlheState {
    pub game: Option<NlheGame>,
    pub pending_chance: bool,
    pub action_history: Vec<String>,
    pub action_actors: Vec<u8>,
}

#[derive(Clone)]
struct StreetClusters {
    centroids: Vec<Vec<f64>>,
}

#[derive(Clone)]
struct LoadedClusters {
    flop: StreetClusters,
    turn: StreetClusters,
    river: StreetClusters,
}

impl LoadedClusters {
    fn load(cluster_dir: &Path) -> Result<Self, NlheModelError> {
        Ok(Self {
            flop: load_street_clusters(cluster_dir, BucketStreet::Flop)?,
            turn: load_street_clusters(cluster_dir, BucketStreet::Turn)?,
            river: load_street_clusters(cluster_dir, BucketStreet::River)?,
        })
    }

    fn for_bucket_street(&self, street: BucketStreet) -> &StreetClusters {
        match street {
            BucketStreet::Flop => &self.flop,
            BucketStreet::Turn => &self.turn,
            BucketStreet::River => &self.river,
            BucketStreet::Preflop => {
                unreachable!("preflop card bucketing is deterministic and does not use centroids")
            }
        }
    }
}

#[derive(Debug)]
pub enum NlheModelError {
    EmptyDeckSeeds,
    InvalidPlayerCount(usize),
    ClusterLoadIo {
        street: BucketStreet,
        path: PathBuf,
        error: String,
    },
    EmptyCentroids {
        street: BucketStreet,
        path: PathBuf,
    },
    InvalidCentroidDimensions {
        street: BucketStreet,
        path: PathBuf,
        expected: usize,
        actual: usize,
    },
}

impl Display for NlheModelError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyDeckSeeds => write!(f, "deck_seeds cannot be empty"),
            Self::InvalidPlayerCount(count) => {
                write!(f, "NlheGameModel requires at least two players, got {count}")
            }
            Self::ClusterLoadIo {
                street,
                path,
                error,
            } => write!(
                f,
                "failed to load {:?} clustering from {}: {error}",
                street,
                path.display()
            ),
            Self::EmptyCentroids { street, path } => write!(
                f,
                "{:?} clustering at {} has no centroids",
                street,
                path.display()
            ),
            Self::InvalidCentroidDimensions {
                street,
                path,
                expected,
                actual,
            } => write!(
                f,
                "{:?} clustering at {} has incompatible centroid dimension: expected {expected}, got {actual}",
                street,
                path.display()
            ),
        }
    }
}

impl std::error::Error for NlheModelError {}

#[derive(Clone)]
pub struct IndexedAction {
    pub engine_action: Action,
    pub action_token: String,
    pub sort_group: u8,
    pub sort_amount: u32,
    pub policy_slot: u8,
}

pub struct NlheGameModel {
    pub config: NlheConfig,
    pub deck_seeds: Vec<u64>,
    clusters: LoadedClusters,
    equity_cache: DashMap<u64, f64>,
    board_texture_cache: DashMap<u64, u16>,
}

impl NlheGameModel {
    pub fn new(
        config: NlheConfig,
        deck_seeds: Vec<u64>,
        cluster_dir: impl AsRef<Path>,
    ) -> Result<Self, NlheModelError> {
        if deck_seeds.is_empty() {
            return Err(NlheModelError::EmptyDeckSeeds);
        }
        if config.num_players < 2 {
            return Err(NlheModelError::InvalidPlayerCount(config.num_players));
        }
        let clusters = LoadedClusters::load(cluster_dir.as_ref())?;
        Ok(Self {
            config,
            deck_seeds,
            clusters,
            equity_cache: DashMap::new(),
            board_texture_cache: DashMap::new(),
        })
    }

    pub fn heads_up_default(cluster_dir: impl AsRef<Path>) -> Result<Self, NlheModelError> {
        Self::new(
            NlheConfig {
                num_players: 2,
                starting_stack: 2_000,
                small_blind: 10,
                big_blind: 20,
            },
            vec![11, 23, 37, 41, 53, 67, 79, 83],
            cluster_dir,
        )
    }
}

impl GameModel for NlheGameModel {
    type State = NlheState;

    fn root_state(&self) -> Self::State {
        NlheState {
            game: None,
            pending_chance: true,
            action_history: Vec::new(),
            action_actors: Vec::new(),
        }
    }

    fn node_kind(&self, state: &Self::State) -> NodeKind {
        if state.pending_chance {
            return NodeKind::Chance;
        }
        let game = state
            .game
            .as_ref()
            .expect("non-chance state must have an NLHE game");
        if game.is_complete() {
            NodeKind::Terminal
        } else {
            NodeKind::Player(
                game.current_actor()
                    .expect("incomplete game must have current actor"),
            )
        }
    }

    fn legal_action_count(&self, state: &Self::State) -> usize {
        if state.pending_chance {
            return self.deck_seeds.len();
        }
        let game = state
            .game
            .as_ref()
            .expect("non-chance state must have an NLHE game");
        if game.is_complete() {
            return 0;
        }
        let actor = game
            .current_actor()
            .expect("incomplete game must have current actor");
        enumerate_action_space(game, actor).len()
    }

    fn next_state(&self, state: &Self::State, action_idx: usize) -> Self::State {
        if state.pending_chance {
            let seed = self.deck_seeds[action_idx];
            let game = NlheGame::new(self.config, seed)
                .unwrap_or_else(|err| panic!("failed to initialize NLHE game: {err:?}"));
            return NlheState {
                game: Some(game),
                pending_chance: false,
                action_history: Vec::new(),
                action_actors: Vec::new(),
            };
        }

        let mut next = state.clone();
        let game = next
            .game
            .as_mut()
            .expect("non-chance state must have an NLHE game");
        let actor = game
            .current_actor()
            .expect("incomplete game must have current actor");
        let action_space = enumerate_action_space(game, actor);
        let picked = action_space
            .get(action_idx)
            .unwrap_or_else(|| panic!("invalid action index {action_idx} for actor {actor}"));

        game.apply_action(actor, picked.engine_action)
            .unwrap_or_else(|err| panic!("failed to apply legal action: {err:?}"));
        next.action_history.push(picked.action_token.clone());
        next.action_actors.push(actor as u8);
        next
    }

    fn terminal_utility(&self, state: &Self::State, player: usize) -> f64 {
        let game = state
            .game
            .as_ref()
            .expect("terminal utility requires resolved game state");
        if !game.is_complete() {
            return 0.0;
        }
        let player_stack = game.players[player].stack as f64;
        let start_stack = self.config.starting_stack as f64;
        (player_stack - start_stack) / self.config.big_blind as f64
    }

    fn infoset_key(&self, state: &Self::State, player: usize) -> String {
        let game = state
            .game
            .as_ref()
            .expect("infosets only exist after chance sampling");
        let player_state = &game.players[player];
        let card_bucket = card_bucket_for_player(
            game,
            player,
            player_state,
            &self.clusters,
            &self.equity_cache,
        );
        let board_bucket = board_texture_bucket(game, &self.board_texture_cache);
        let action_history_hash = hash_value(&(
            state.action_history.as_slice(),
            state.action_actors.as_slice(),
        ));

        format!(
            "p={player}|street={}|cb={card_bucket}|bb={board_bucket}|h={action_history_hash}",
            street_code(game.round),
        )
    }

    fn chance_probabilities(&self, state: &Self::State) -> Vec<f64> {
        if state.pending_chance {
            vec![1.0 / self.deck_seeds.len() as f64; self.deck_seeds.len()]
        } else {
            Vec::new()
        }
    }
}

fn street_code(round: BettingRound) -> &'static str {
    match round {
        BettingRound::Preflop => "P",
        BettingRound::Flop => "F",
        BettingRound::Turn => "T",
        BettingRound::River => "R",
        BettingRound::Complete => "X",
    }
}

fn round_to_abstract_street(round: BettingRound) -> AbstractStreet {
    match round {
        BettingRound::Preflop => AbstractStreet::Preflop,
        BettingRound::Flop => AbstractStreet::Flop,
        BettingRound::Turn => AbstractStreet::Turn,
        BettingRound::River | BettingRound::Complete => AbstractStreet::River,
    }
}

fn round_to_bucket_street(round: BettingRound) -> BucketStreet {
    match round {
        BettingRound::Preflop => BucketStreet::Preflop,
        BettingRound::Flop => BucketStreet::Flop,
        BettingRound::Turn => BucketStreet::Turn,
        BettingRound::River | BettingRound::Complete => BucketStreet::River,
    }
}

fn cluster_filename(street: BucketStreet) -> &'static str {
    match street {
        BucketStreet::Preflop => "preflop.clusters",
        BucketStreet::Flop => "flop.clusters",
        BucketStreet::Turn => "turn.clusters",
        BucketStreet::River => "river.clusters",
    }
}

fn load_street_clusters(
    cluster_dir: &Path,
    street: BucketStreet,
) -> Result<StreetClusters, NlheModelError> {
    let path = cluster_dir.join(cluster_filename(street));
    let clustering =
        load_clustering_result(&path).map_err(|err| NlheModelError::ClusterLoadIo {
            street,
            path: path.clone(),
            error: err.to_string(),
        })?;
    validate_clustering(&clustering, street, &path)?;
    Ok(StreetClusters {
        centroids: clustering.centroids,
    })
}

fn validate_clustering(
    clustering: &ClusteringResult,
    street: BucketStreet,
    path: &Path,
) -> Result<(), NlheModelError> {
    if clustering.centroids.is_empty() {
        return Err(NlheModelError::EmptyCentroids {
            street,
            path: path.to_path_buf(),
        });
    }
    let expected_dims = clustering.centroids[0].len();
    if expected_dims == 0 {
        return Err(NlheModelError::InvalidCentroidDimensions {
            street,
            path: path.to_path_buf(),
            expected: 1,
            actual: 0,
        });
    }
    for centroid in &clustering.centroids {
        if centroid.len() != expected_dims {
            return Err(NlheModelError::InvalidCentroidDimensions {
                street,
                path: path.to_path_buf(),
                expected: expected_dims,
                actual: centroid.len(),
            });
        }
    }
    let expected_bucket_count = buckets_for_street(street);
    if clustering.centroids.len() != expected_bucket_count {
        return Err(NlheModelError::InvalidCentroidDimensions {
            street,
            path: path.to_path_buf(),
            expected: expected_bucket_count,
            actual: clustering.centroids.len(),
        });
    }
    Ok(())
}

fn hash_value<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn assign_bucket(hist: &[f64], clusters: &StreetClusters) -> u16 {
    if clusters.centroids.is_empty() || clusters.centroids[0].len() != hist.len() {
        return 0;
    }
    clusters
        .centroids
        .iter()
        .enumerate()
        .map(|(idx, centroid)| (idx, emd_distance(hist, centroid)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        .map(|(idx, _)| idx as u16)
        .unwrap_or(0)
}

fn card_bucket_for_player(
    game: &NlheGame,
    player_idx: usize,
    player_state: &PlayerState,
    clusters: &LoadedClusters,
    equity_cache: &DashMap<u64, f64>,
) -> u16 {
    let street = round_to_bucket_street(game.round);
    if street == BucketStreet::Preflop {
        let hole = [
            to_rs_card(player_state.hole_cards[0]),
            to_rs_card(player_state.hole_cards[1]),
        ];
        let canonical = canonicalize_preflop_hand(hole);
        return canonical_preflop_index(canonical) as u16;
    }

    let cache_key = hash_value(&(
        player_state.hole_cards,
        &game.board,
        street_code(game.round),
        game.players
            .iter()
            .map(|p| (p.folded, p.all_in))
            .collect::<Vec<_>>(),
    ));
    let equity = if let Some(v) = equity_cache.get(&cache_key) {
        *v
    } else {
        let hole = [
            to_rs_card(player_state.hole_cards[0]),
            to_rs_card(player_state.hole_cards[1]),
        ];
        let board = game
            .board
            .iter()
            .copied()
            .map(to_rs_card)
            .collect::<Vec<_>>();
        let opponent_count = game
            .players
            .iter()
            .enumerate()
            .filter(|(idx, p)| *idx != player_idx && !p.folded)
            .count()
            .max(1);
        let seed = cache_key ^ 0xA3B1_C2D3_E4F5_6789;
        let computed = compute_equity_multiway_with_seed(
            hole,
            &board,
            opponent_count,
            RUNTIME_EQUITY_SAMPLES,
            seed,
        );
        equity_cache.insert(cache_key, computed);
        computed
    };
    let hist = equity_to_histogram(equity);
    debug_assert_eq!(hist.len(), EQUITY_DIMS);
    assign_bucket(&hist, clusters.for_bucket_street(street))
}

fn board_suit_pressure_bucket(board: &[game::Card]) -> u16 {
    let mut suit_counts = [0u8; 4];
    for card in board {
        let suit_idx = match card.suit {
            game::Suit::Clubs => 0usize,
            game::Suit::Diamonds => 1usize,
            game::Suit::Hearts => 2usize,
            game::Suit::Spades => 3usize,
        };
        suit_counts[suit_idx] = suit_counts[suit_idx].saturating_add(1);
    }
    let max_suit = suit_counts.into_iter().max().unwrap_or(0);
    u16::from(max_suit >= 3)
}

fn board_texture_bucket(game: &NlheGame, board_texture_cache: &DashMap<u64, u16>) -> u16 {
    if game.board.is_empty() {
        return 0;
    }
    let active_players = game.players.iter().filter(|p| !p.folded).count().max(2);
    let opponent_count = active_players.saturating_sub(1).max(1);
    let cache_key = hash_value(&(street_code(game.round), game.board.as_slice(), opponent_count));
    if let Some(bucket) = board_texture_cache.get(&cache_key) {
        return *bucket;
    }

    let board = game.board.iter().copied().map(to_rs_card).collect::<Vec<_>>();
    let seed = cache_key ^ 0x5BAA_9D17_3E6C_C2F1;
    let board_equity = compute_board_texture_equity_multiway_with_seed(
        &board,
        opponent_count,
        RUNTIME_BOARD_TEXTURE_SAMPLES,
        seed,
    );
    let equity_hist = equity_to_histogram(board_equity);
    debug_assert_eq!(equity_hist.len(), EQUITY_DIMS);
    let equity_bin = equity_hist
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        .map(|(idx, _)| idx as u16)
        .unwrap_or(0)
        .min((BOARD_TEXTURE_BUCKETS / 2).saturating_sub(1));
    let suit_pressure = board_suit_pressure_bucket(&game.board);
    let bucket = (equity_bin.saturating_mul(2).saturating_add(suit_pressure))
        .min(BOARD_TEXTURE_BUCKETS.saturating_sub(1));
    board_texture_cache.insert(cache_key, bucket);
    bucket
}

fn to_rs_card(card: game::Card) -> RsCard {
    let value = match card.rank {
        Rank::Two => RsValue::Two,
        Rank::Three => RsValue::Three,
        Rank::Four => RsValue::Four,
        Rank::Five => RsValue::Five,
        Rank::Six => RsValue::Six,
        Rank::Seven => RsValue::Seven,
        Rank::Eight => RsValue::Eight,
        Rank::Nine => RsValue::Nine,
        Rank::Ten => RsValue::Ten,
        Rank::Jack => RsValue::Jack,
        Rank::Queen => RsValue::Queen,
        Rank::King => RsValue::King,
        Rank::Ace => RsValue::Ace,
    };
    let suit = match card.suit {
        game::Suit::Clubs => RsSuit::Club,
        game::Suit::Diamonds => RsSuit::Diamond,
        game::Suit::Hearts => RsSuit::Heart,
        game::Suit::Spades => RsSuit::Spade,
    };
    RsCard::new(value, suit)
}

fn pot_before_action(game: &NlheGame) -> u32 {
    game.players
        .iter()
        .map(|p| p.total_contribution)
        .sum::<u32>()
}

fn abstract_action_token(abstract_action: AbstractAction) -> String {
    match abstract_action {
        AbstractAction::Fold => "fold".to_string(),
        AbstractAction::Check => "check".to_string(),
        AbstractAction::Call => "call".to_string(),
        AbstractAction::AllIn => "allin".to_string(),
        AbstractAction::BetPotFraction(v) => format!("bet_{v:.2}"),
        AbstractAction::RaiseMultiplier(v) => format!("raise_{v:.2}"),
        AbstractAction::OpenRaiseBbMultiple(v) => format!("open_{v:.2}"),
        AbstractAction::FourBetMultiplier(v) => format!("4bet_{v:.2}"),
    }
}

fn preflop_reraise_family(preflop_raise_count: u8) -> String {
    // preflop_raise_count tracks aggressive raises excluding forced blinds.
    // open => 1, next raise => 3-bet, then 4-bet, ...
    let n_bet = preflop_raise_count.saturating_add(2);
    format!("{n_bet}bet")
}

fn action_identity(action: Action) -> String {
    match action {
        Action::Fold => "f".to_string(),
        Action::Check => "x".to_string(),
        Action::Call => "c".to_string(),
        Action::AllIn => "ai".to_string(),
        Action::Bet(total) => format!("bet:{total}"),
        Action::RaiseTo(total) => format!("raise:{total}"),
    }
}

fn clamp_total(target: u32, min: u32, max: u32) -> u32 {
    target.max(min).min(max)
}

fn push_unique_action(
    out: &mut Vec<IndexedAction>,
    seen: &mut HashSet<String>,
    action: Action,
    token: String,
    sort_group: u8,
    sort_amount: u32,
) {
    let id = action_identity(action);
    if seen.insert(id) {
        out.push(IndexedAction {
            engine_action: action,
            action_token: token,
            sort_group,
            sort_amount,
            policy_slot: 0,
        });
    }
}

fn assign_policy_slots(actions: &mut [IndexedAction]) {
    let mut bet_rank = 0usize;
    let mut raise_rank = 0usize;
    for action in actions.iter_mut() {
        action.policy_slot = match action.sort_group {
            0 => 0, // fold
            1 => 1, // check
            2 => 2, // call
            3 => {
                let slot = POLICY_BET_SLOT_START + bet_rank.min(POLICY_BET_SLOT_COUNT - 1) as u8;
                bet_rank += 1;
                slot
            }
            4 => {
                let slot =
                    POLICY_RAISE_SLOT_START + raise_rank.min(POLICY_RAISE_SLOT_COUNT - 1) as u8;
                raise_rank += 1;
                slot
            }
            5 => POLICY_ALL_IN_SLOT, // all-in
            _ => POLICY_ALL_IN_SLOT,
        };
    }
    debug_assert_eq!(
        actions
            .iter()
            .map(|a| a.policy_slot)
            .collect::<HashSet<_>>()
            .len(),
        actions.len(),
        "policy slots should be unique per action-space"
    );
    debug_assert!(
        actions
            .iter()
            .all(|a| usize::from(a.policy_slot) < POLICY_MAX_ACTIONS),
        "policy slot id exceeded POLICY_MAX_ACTIONS"
    );
}

pub fn enumerate_action_space(game: &NlheGame, actor: usize) -> Vec<IndexedAction> {
    let legal = game
        .legal_actions(actor)
        .expect("failed to fetch legal actions");
    let actor_state = &game.players[actor];
    let street = round_to_abstract_street(game.round);
    let pot_before = pot_before_action(game).max(1) as f64;
    let to_call = legal
        .iter()
        .find_map(|action| match action {
            LegalAction::Call { amount } => Some(*amount),
            _ => None,
        })
        .unwrap_or(0);

    let mut out = Vec::new();
    let mut seen = HashSet::new();

    for legal_action in legal {
        match legal_action {
            LegalAction::Fold => {
                push_unique_action(&mut out, &mut seen, Action::Fold, "f".to_string(), 0, 0)
            }
            LegalAction::Check => {
                push_unique_action(&mut out, &mut seen, Action::Check, "x".to_string(), 1, 0)
            }
            LegalAction::Call { .. } => push_unique_action(
                &mut out,
                &mut seen,
                Action::Call,
                "c".to_string(),
                2,
                to_call,
            ),
            LegalAction::Bet { min, max } => match street {
                AbstractStreet::Preflop => {
                    for &bb_multiple in preflop_open_raise_grid() {
                        let target = (bb_multiple * game.config.big_blind as f64)
                            .round()
                            .max(1.0) as u32;
                        let total = clamp_total(target, min, max);
                        let token = format!(
                            "b:{}",
                            abstract_action_token(AbstractAction::OpenRaiseBbMultiple(bb_multiple))
                        );
                        push_unique_action(
                            &mut out,
                            &mut seen,
                            Action::Bet(total),
                            token,
                            3,
                            total,
                        );
                    }
                }
                _ => {
                    for &fraction in street_bet_grid(street) {
                        let delta = (fraction * pot_before).round().max(1.0) as u32;
                        let target = actor_state.street_contribution.saturating_add(delta);
                        let total = clamp_total(target, min, max);
                        let token = format!(
                            "b:{}",
                            abstract_action_token(AbstractAction::BetPotFraction(fraction))
                        );
                        push_unique_action(
                            &mut out,
                            &mut seen,
                            Action::Bet(total),
                            token,
                            3,
                            total,
                        );
                    }
                }
            },
            LegalAction::RaiseTo { min, max } => match street {
                AbstractStreet::Preflop => {
                    let current_bet = actor_state.street_contribution.saturating_add(to_call);
                    let family = preflop_reraise_family(game.preflop_raise_count());
                    for &multiplier in preflop_reraise_multiplier_grid() {
                        let target = (current_bet as f64 * multiplier).round().max(1.0) as u32;
                        let total = clamp_total(target, min, max);
                        let token = format!("r:{family}_{multiplier:.2}x");
                        push_unique_action(
                            &mut out,
                            &mut seen,
                            Action::RaiseTo(total),
                            token,
                            4,
                            total,
                        );
                    }
                }
                _ => {
                    let raise_grid = street_raise_multiplier_grid(street);
                    if !raise_grid.is_empty() {
                        let current_bet = actor_state.street_contribution.saturating_add(to_call);
                        let reference = to_call.max(game.config.big_blind);
                        for &multiplier in raise_grid {
                            let raise_delta =
                                (reference as f64 * multiplier).round().max(1.0) as u32;
                            let target = current_bet.saturating_add(raise_delta);
                            let total = clamp_total(target, min, max);
                            let token = format!(
                                "r:{}",
                                abstract_action_token(AbstractAction::RaiseMultiplier(multiplier))
                            );
                            push_unique_action(
                                &mut out,
                                &mut seen,
                                Action::RaiseTo(total),
                                token,
                                4,
                                total,
                            );
                        }
                    } else {
                        let delta = min.saturating_sub(actor_state.street_contribution) as f64;
                        let mapped = map_real_bet_to_abstract(street, delta, pot_before);
                        let token = format!("r:{}", abstract_action_token(mapped));
                        push_unique_action(
                            &mut out,
                            &mut seen,
                            Action::RaiseTo(min),
                            token,
                            4,
                            min,
                        );
                    }
                }
            },
            LegalAction::AllIn { .. } => {
                let total = actor_state
                    .street_contribution
                    .saturating_add(actor_state.stack);
                push_unique_action(
                    &mut out,
                    &mut seen,
                    Action::AllIn,
                    "ai".to_string(),
                    5,
                    total,
                );
            }
        }
    }

    out.sort_by(|a, b| {
        a.sort_group
            .cmp(&b.sort_group)
            .then(a.sort_amount.cmp(&b.sort_amount))
            .then(a.action_token.cmp(&b.action_token))
    });
    assign_policy_slots(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use abstraction::clustering::{
        buckets_for_street, save_clustering_result, BucketStreet, ClusteringResult,
    };
    use abstraction::EQUITY_DIMS;
    use game::{Action, BettingRound, Card, Rank, Suit};

    use crate::external_sampling::ExternalSamplingTrainer;
    use crate::external_sampling::GameModel;
    use crate::nlhe_game::{
        cluster_filename, enumerate_action_space, NlheGameModel, NlheModelError,
    };

    static UNIQUE_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn now_nanos() -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    }

    fn unique_temp_suffix() -> String {
        let ctr = UNIQUE_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
        format!("{}_{}_{}", now_nanos(), process::id(), ctr)
    }

    fn write_test_clusters(cluster_dir: &Path) {
        fs::create_dir_all(cluster_dir).expect("cluster dir should be created");
        for street in [
            BucketStreet::Preflop,
            BucketStreet::Flop,
            BucketStreet::Turn,
            BucketStreet::River,
        ] {
            let k = buckets_for_street(street);
            let dims = EQUITY_DIMS;
            let centroids = (0..k)
                .map(|idx| {
                    let mut hist = vec![0.0; dims];
                    hist[idx % dims] = 1.0;
                    hist
                })
                .collect::<Vec<_>>();
            let result = ClusteringResult {
                assignments: vec![0; k],
                centroids,
                iterations_run: 0,
            };
            let path = cluster_dir.join(cluster_filename(street));
            save_clustering_result(&path, &result).expect("cluster file should be saved");
        }
    }

    fn make_model_with_temp_clusters() -> (NlheGameModel, PathBuf) {
        let dir = env::temp_dir().join(format!("talibus_nlhe_clusters_{}", unique_temp_suffix()));
        write_test_clusters(&dir);
        let model =
            NlheGameModel::heads_up_default(&dir).expect("cluster-backed model should load");
        (model, dir)
    }

    #[test]
    fn model_fails_fast_when_cluster_files_are_missing() {
        let dir = env::temp_dir().join(format!(
            "talibus_nlhe_missing_clusters_{}",
            unique_temp_suffix()
        ));
        fs::create_dir_all(&dir).expect("temp dir");
        let result = NlheGameModel::heads_up_default(&dir);
        fs::remove_dir_all(&dir).ok();

        assert!(matches!(result, Err(NlheModelError::ClusterLoadIo { .. })));
    }

    #[test]
    fn hu_nlhe_model_runs_serial_mccfr_iterations() {
        let (model, dir) = make_model_with_temp_clusters();
        let mut trainer = ExternalSamplingTrainer::new(2);
        trainer.train_serial(&model, 20, 42);
        assert!(trainer.infoset_count() > 0);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn hu_nlhe_model_runs_parallel_mccfr_iterations() {
        let (model, dir) = make_model_with_temp_clusters();
        let mut trainer = ExternalSamplingTrainer::new(2);
        trainer.train_parallel(&model, 20, 2, 99);
        assert!(trainer.infoset_count() > 0);
        let (total_infosets, non_uniform_infosets, avg_actions, max_actions) =
            trainer.diagnostic_info();
        assert!(total_infosets > 0);
        assert!(non_uniform_infosets > 0);
        assert!(avg_actions.is_finite());
        assert!(max_actions > 0);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn legal_action_count_matches_abstract_action_space() {
        let (model, dir) = make_model_with_temp_clusters();
        let root = model.root_state();
        let state = model.next_state(&root, 0);
        let game = state.game.as_ref().expect("chance should deal a hand");
        let actor = game.current_actor().expect("actor expected");
        let action_space = enumerate_action_space(game, actor);
        assert_eq!(model.legal_action_count(&state), action_space.len());
        assert!(!action_space.is_empty());
        assert!(action_space
            .iter()
            .any(|a| a.action_token.starts_with("r:")));
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn preflop_reraise_space_has_multiple_sizes() {
        let (model, dir) = make_model_with_temp_clusters();
        let root = model.root_state();
        let mut state = model.next_state(&root, 0);
        let game = state
            .game
            .as_mut()
            .expect("chance node should produce a game state");

        let opener = game.current_actor().expect("preflop should have an actor");
        let open_action = enumerate_action_space(game, opener)
            .into_iter()
            .find_map(|a| match a.engine_action {
                Action::RaiseTo(total) if total > game.config.big_blind => Some(a.engine_action),
                _ => None,
            })
            .expect("preflop opener should have at least one raise option");
        game.apply_action(opener, open_action)
            .expect("open action should be legal");

        let responder = game
            .current_actor()
            .expect("responder should act after open");
        let mut raise_totals = enumerate_action_space(game, responder)
            .into_iter()
            .filter_map(|a| match a.engine_action {
                Action::RaiseTo(total) => Some(total),
                _ => None,
            })
            .collect::<Vec<_>>();
        raise_totals.sort_unstable();
        raise_totals.dedup();

        assert!(
            raise_totals.len() >= 2,
            "expected at least 2 distinct preflop re-raise sizes, got {raise_totals:?}"
        );
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn preflop_reraise_tokens_keep_raise_depth_distinct() {
        let (model, dir) = make_model_with_temp_clusters();
        let root = model.root_state();
        let mut state = model.next_state(&root, 0);
        let game = state
            .game
            .as_mut()
            .expect("chance node should produce a game state");

        let opener = game.current_actor().expect("preflop should have an actor");
        let open_action = enumerate_action_space(game, opener)
            .into_iter()
            .find_map(|a| match a.engine_action {
                Action::RaiseTo(_) => Some(a.engine_action),
                _ => None,
            })
            .expect("preflop opener should have at least one raise option");
        game.apply_action(opener, open_action)
            .expect("open action should be legal");

        let responder = game
            .current_actor()
            .expect("responder should act after open");
        let reraise_space = enumerate_action_space(game, responder);
        let three_bet_tokens = reraise_space
            .iter()
            .filter_map(|a| match a.engine_action {
                Action::RaiseTo(_) => Some(a.action_token.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(
            !three_bet_tokens.is_empty(),
            "responder should see preflop re-raise options"
        );
        assert!(
            three_bet_tokens.iter().all(|t| t.starts_with("r:3bet_")),
            "expected responder re-raise tokens to be 3-bet family, got {three_bet_tokens:?}"
        );

        let selected_three_bet = reraise_space
            .into_iter()
            .find_map(|a| match a.engine_action {
                Action::RaiseTo(_) => Some(a.engine_action),
                _ => None,
            })
            .expect("responder should have at least one 3-bet");
        game.apply_action(responder, selected_three_bet)
            .expect("3-bet action should be legal");

        let back_to_opener = game.current_actor().expect("opener should act after 3-bet");
        let four_bet_tokens = enumerate_action_space(game, back_to_opener)
            .into_iter()
            .filter_map(|a| match a.engine_action {
                Action::RaiseTo(_) => Some(a.action_token),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(
            !four_bet_tokens.is_empty(),
            "opener should see re-raise options after facing a 3-bet"
        );
        assert!(
            four_bet_tokens.iter().all(|t| t.starts_with("r:4bet_")),
            "expected opener re-raise tokens to be 4-bet family, got {four_bet_tokens:?}"
        );
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn preflop_bucket_separates_aks_and_ako() {
        let (model, dir) = make_model_with_temp_clusters();
        let mut game = game::NlheGame::new(model.config, 42).expect("game should initialize");
        let player_idx = 0usize;

        game.players[player_idx].hole_cards = [
            Card {
                rank: Rank::Ace,
                suit: Suit::Spades,
            },
            Card {
                rank: Rank::King,
                suit: Suit::Spades,
            },
        ];
        let suited_player = game.players[player_idx].clone();
        let suited_bucket = super::card_bucket_for_player(
            &game,
            player_idx,
            &suited_player,
            &model.clusters,
            &model.equity_cache,
        );

        game.players[player_idx].hole_cards = [
            Card {
                rank: Rank::Ace,
                suit: Suit::Spades,
            },
            Card {
                rank: Rank::King,
                suit: Suit::Hearts,
            },
        ];
        let offsuit_player = game.players[player_idx].clone();
        let offsuit_bucket = super::card_bucket_for_player(
            &game,
            player_idx,
            &offsuit_player,
            &model.clusters,
            &model.equity_cache,
        );

        assert_ne!(
            suited_bucket, offsuit_bucket,
            "AK suited and AK offsuit must map to distinct canonical preflop buckets"
        );
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn board_texture_bucket_distinguishes_monotone_and_rainbow_flops() {
        let (model, dir) = make_model_with_temp_clusters();
        let root = model.root_state();
        let mut state = model.next_state(&root, 0);
        let game = state
            .game
            .as_mut()
            .expect("chance node should produce a game state");
        game.round = BettingRound::Flop;

        game.board = vec![
            Card {
                rank: Rank::Ace,
                suit: Suit::Spades,
            },
            Card {
                rank: Rank::King,
                suit: Suit::Spades,
            },
            Card {
                rank: Rank::Queen,
                suit: Suit::Spades,
            },
        ];
        let monotone = super::board_texture_bucket(game, &model.board_texture_cache);

        game.board = vec![
            Card {
                rank: Rank::Seven,
                suit: Suit::Clubs,
            },
            Card {
                rank: Rank::Four,
                suit: Suit::Diamonds,
            },
            Card {
                rank: Rank::Two,
                suit: Suit::Hearts,
            },
        ];
        let rainbow = super::board_texture_bucket(game, &model.board_texture_cache);

        assert_ne!(
            monotone, rainbow,
            "materially different flops should not collapse to the same board bucket"
        );
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn expanded_slot_layout_matches_expected_solver_ordering() {
        let (model, dir) = make_model_with_temp_clusters();
        let root = model.root_state();
        let mut state = model.next_state(&root, 0);
        let game = state
            .game
            .as_mut()
            .expect("chance node should produce a game state");

        let opener = game.current_actor().expect("preflop should have an actor");
        let open_action = enumerate_action_space(game, opener)
            .into_iter()
            .find_map(|a| match a.engine_action {
                Action::RaiseTo(total) if total > game.config.big_blind => Some(a.engine_action),
                _ => None,
            })
            .expect("opener should have a raise action");
        game.apply_action(opener, open_action)
            .expect("open action should be legal");

        let responder = game
            .current_actor()
            .expect("responder should act after open");
        let preflop_slots = enumerate_action_space(game, responder)
            .iter()
            .map(|a| a.policy_slot)
            .collect::<Vec<_>>();
        assert_eq!(
            preflop_slots,
            vec![0, 2, 7, 8, 9, 10, 11],
            "preflop facing-raise slot ordering should match expanded schema"
        );

        let mut flop_state = model.next_state(&root, 1);
        let game_for_flop = flop_state
            .game
            .as_mut()
            .expect("chance node should produce a game state");
        let preflop_first_actor = game_for_flop
            .current_actor()
            .expect("preflop should have first actor");
        game_for_flop
            .apply_action(preflop_first_actor, Action::Call)
            .expect("limp/call should be legal");
        let preflop_second_actor = game_for_flop
            .current_actor()
            .expect("preflop should have second actor");
        game_for_flop
            .apply_action(preflop_second_actor, Action::Check)
            .expect("check should close preflop after limp");

        assert_eq!(game_for_flop.round, BettingRound::Flop);
        let flop_actor = game_for_flop.current_actor().expect("flop should have an actor");
        let flop_slots = enumerate_action_space(game_for_flop, flop_actor)
            .iter()
            .map(|a| a.policy_slot)
            .collect::<Vec<_>>();
        assert_eq!(
            flop_slots,
            vec![1, 3, 4, 5, 6],
            "flop no-bet slot ordering should match expanded schema"
        );
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn action_space_policy_slots_are_unique() {
        let (model, dir) = make_model_with_temp_clusters();
        let root = model.root_state();
        let state = model.next_state(&root, 0);
        let game = state.game.as_ref().expect("chance should deal a hand");
        let actor = game.current_actor().expect("actor expected");
        let action_space = enumerate_action_space(game, actor);
        assert!(!action_space.is_empty(), "action space should not be empty");
        let mut slots = action_space
            .iter()
            .map(|a| a.policy_slot)
            .collect::<Vec<_>>();
        slots.sort_unstable();
        slots.dedup();
        assert_eq!(
            slots.len(),
            action_space.len(),
            "policy slots should be unique per state"
        );
        fs::remove_dir_all(dir).ok();
    }
}
