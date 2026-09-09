use cfr::nlhe_game::NlheState;
use game::{BettingRound, Card, LegalAction, NlheGame, Rank, SeatPosition, Suit};

pub const MAX_PLAYERS: usize = 6;
pub const HISTORY_SLOTS: usize = 24;
pub const HISTORY_ACTION_BUCKET_DIMS: usize = 8;
pub const HISTORY_ACTOR_DIMS: usize = MAX_PLAYERS;
pub const HISTORY_ACTION_DIMS: usize = HISTORY_ACTION_BUCKET_DIMS + HISTORY_ACTOR_DIMS + 1;
pub const INPUT_DIM: usize = 510;

const CARD_DIMS: usize = 52;
const STREET_DIMS: usize = 4;
const POSITION_DIMS: usize = 6;
const PLAYER_VECTOR_DIMS: usize = MAX_PLAYERS;

const HOLE_OFFSET: usize = 0;
const BOARD_OFFSET: usize = HOLE_OFFSET + CARD_DIMS;
const STREET_OFFSET: usize = BOARD_OFFSET + CARD_DIMS;
const HERO_POSITION_OFFSET: usize = STREET_OFFSET + STREET_DIMS;
const ACTOR_POSITION_OFFSET: usize = HERO_POSITION_OFFSET + POSITION_DIMS;
const POT_OFFSET: usize = ACTOR_POSITION_OFFSET + POSITION_DIMS;
const TO_CALL_OFFSET: usize = POT_OFFSET + 1;
const HERO_STACK_OFFSET: usize = TO_CALL_OFFSET + 1;
const HERO_STREET_CONTRIB_OFFSET: usize = HERO_STACK_OFFSET + 1;
const ACTIVE_PLAYERS_RATIO_OFFSET: usize = HERO_STREET_CONTRIB_OFFSET + 1;
const PLAYER_ACTIVE_MASK_OFFSET: usize = ACTIVE_PLAYERS_RATIO_OFFSET + 1;
const PLAYER_STACK_OFFSET: usize = PLAYER_ACTIVE_MASK_OFFSET + PLAYER_VECTOR_DIMS;
const PLAYER_TOTAL_CONTRIB_OFFSET: usize = PLAYER_STACK_OFFSET + PLAYER_VECTOR_DIMS;
const PLAYER_STREET_CONTRIB_OFFSET: usize = PLAYER_TOTAL_CONTRIB_OFFSET + PLAYER_VECTOR_DIMS;
const PREFLOP_RAISE_COUNT_OFFSET: usize = PLAYER_STREET_CONTRIB_OFFSET + PLAYER_VECTOR_DIMS;
const HISTORY_OFFSET: usize = PREFLOP_RAISE_COUNT_OFFSET + 1;

const ACTION_BUCKET_FOLD: usize = 0;
const ACTION_BUCKET_CHECK: usize = 1;
const ACTION_BUCKET_CALL: usize = 2;
const ACTION_BUCKET_BET_SMALL: usize = 3;
const ACTION_BUCKET_BET_MEDIUM: usize = 4;
const ACTION_BUCKET_BET_LARGE: usize = 5;
const ACTION_BUCKET_RAISE: usize = 6;
const ACTION_BUCKET_ALL_IN: usize = 7;

pub fn encode_nlhe_state(state: &NlheState, player_idx: usize) -> Option<[f32; INPUT_DIM]> {
    let game = state.game.as_ref()?;
    Some(encode_nlhe_game(
        game,
        player_idx,
        &state.action_history,
        &state.action_actors,
    ))
}

pub fn encode_nlhe_game(
    game: &NlheGame,
    player_idx: usize,
    action_history: &[String],
    action_actors: &[u8],
) -> [f32; INPUT_DIM] {
    let mut out = [0.0f32; INPUT_DIM];
    if player_idx >= game.players.len() {
        return out;
    }

    let player = &game.players[player_idx];

    // Hole cards multi-hot.
    for &card in &player.hole_cards {
        let idx = card_index(card);
        out[HOLE_OFFSET + idx] = 1.0;
    }

    // Board cards multi-hot.
    for &card in &game.board {
        let idx = card_index(card);
        out[BOARD_OFFSET + idx] = 1.0;
    }

    // Street one-hot (Complete maps to river slot).
    out[STREET_OFFSET + street_index(game.round)] = 1.0;

    // Hero/actor position one-hot.
    out[HERO_POSITION_OFFSET + position_index(player.seat)] = 1.0;
    if let Some(actor_idx) = game.current_actor() {
        if actor_idx < game.players.len() {
            out[ACTOR_POSITION_OFFSET + position_index(game.players[actor_idx].seat)] = 1.0;
        }
    }

    let starting_stack = game.config.starting_stack.max(1) as f32;
    let total_pot = game
        .players
        .iter()
        .map(|p| p.total_contribution as f32)
        .sum::<f32>();
    out[POT_OFFSET] = total_pot / (starting_stack * game.players.len().max(1) as f32).max(1.0);
    out[TO_CALL_OFFSET] = amount_to_call(game, player_idx) / starting_stack;
    out[HERO_STACK_OFFSET] = player.stack as f32 / starting_stack;
    out[HERO_STREET_CONTRIB_OFFSET] = player.street_contribution as f32 / starting_stack;

    let active_players = game.players.iter().filter(|p| !p.folded).count();
    out[ACTIVE_PLAYERS_RATIO_OFFSET] = active_players as f32 / game.players.len().max(1) as f32;

    for (idx, state) in game.players.iter().enumerate().take(MAX_PLAYERS) {
        out[PLAYER_ACTIVE_MASK_OFFSET + idx] = if state.folded { 0.0 } else { 1.0 };
        out[PLAYER_STACK_OFFSET + idx] = state.stack as f32 / starting_stack;
        out[PLAYER_TOTAL_CONTRIB_OFFSET + idx] = state.total_contribution as f32 / starting_stack;
        out[PLAYER_STREET_CONTRIB_OFFSET + idx] = state.street_contribution as f32 / starting_stack;
    }

    out[PREFLOP_RAISE_COUNT_OFFSET] = (game.preflop_raise_count() as f32 / 6.0).clamp(0.0, 1.0);

    // Action history: 24 slots * (8 action buckets + 6 actor slots + 1 amount).
    let start_idx = action_history.len().saturating_sub(HISTORY_SLOTS);
    for (slot, token_idx) in (start_idx..action_history.len()).enumerate() {
        let token = &action_history[token_idx];
        let base = HISTORY_OFFSET + slot * HISTORY_ACTION_DIMS;
        if let Some(bucket) = classify_action(token) {
            out[base + bucket] = 1.0;
        }
        let actor = action_actors.get(token_idx).copied().unwrap_or(u8::MAX) as usize;
        if actor < MAX_PLAYERS {
            out[base + HISTORY_ACTION_BUCKET_DIMS + actor] = 1.0;
        }
        out[base + HISTORY_ACTION_DIMS - 1] = normalized_token_amount(token);
    }

    out
}

fn amount_to_call(game: &NlheGame, player_idx: usize) -> f32 {
    if game.current_actor() != Some(player_idx) {
        return 0.0;
    }
    match game.legal_actions(player_idx) {
        Ok(actions) => actions
            .iter()
            .find_map(|action| match action {
                LegalAction::Call { amount } => Some(*amount as f32),
                _ => None,
            })
            .unwrap_or(0.0),
        Err(_) => 0.0,
    }
}

fn street_index(round: BettingRound) -> usize {
    match round {
        BettingRound::Preflop => 0,
        BettingRound::Flop => 1,
        BettingRound::Turn => 2,
        BettingRound::River | BettingRound::Complete => 3,
    }
}

fn position_index(seat: SeatPosition) -> usize {
    match seat {
        SeatPosition::UnderTheGun => 0,
        SeatPosition::Hijack => 1,
        SeatPosition::Cutoff => 2,
        SeatPosition::Button => 3,
        SeatPosition::SmallBlind => 4,
        SeatPosition::BigBlind => 5,
    }
}

fn card_index(card: Card) -> usize {
    let rank_idx = match card.rank {
        Rank::Two => 0,
        Rank::Three => 1,
        Rank::Four => 2,
        Rank::Five => 3,
        Rank::Six => 4,
        Rank::Seven => 5,
        Rank::Eight => 6,
        Rank::Nine => 7,
        Rank::Ten => 8,
        Rank::Jack => 9,
        Rank::Queen => 10,
        Rank::King => 11,
        Rank::Ace => 12,
    };
    let suit_idx = match card.suit {
        Suit::Clubs => 0,
        Suit::Diamonds => 1,
        Suit::Hearts => 2,
        Suit::Spades => 3,
    };
    rank_idx * 4 + suit_idx
}

fn classify_action(token: &str) -> Option<usize> {
    if token == "f" {
        return Some(ACTION_BUCKET_FOLD);
    }
    if token == "x" {
        return Some(ACTION_BUCKET_CHECK);
    }
    if token == "c" {
        return Some(ACTION_BUCKET_CALL);
    }
    if token == "ai" {
        return Some(ACTION_BUCKET_ALL_IN);
    }
    if token.starts_with("r:") {
        return Some(ACTION_BUCKET_RAISE);
    }
    if token.starts_with("b:") {
        let amount = normalized_token_amount(token);
        return if amount <= 0.33 {
            Some(ACTION_BUCKET_BET_SMALL)
        } else if amount <= 0.66 {
            Some(ACTION_BUCKET_BET_MEDIUM)
        } else {
            Some(ACTION_BUCKET_BET_LARGE)
        };
    }
    None
}

fn normalized_token_amount(token: &str) -> f32 {
    let parsed = extract_last_number(token).unwrap_or(0.0);
    (parsed / 4.0).clamp(0.0, 1.0)
}

fn extract_last_number(token: &str) -> Option<f32> {
    let mut current = String::new();
    let mut numbers = Vec::new();
    for ch in token.chars() {
        if ch.is_ascii_digit() || ch == '.' {
            current.push(ch);
        } else if !current.is_empty() {
            numbers.push(current.clone());
            current.clear();
        }
    }
    if !current.is_empty() {
        numbers.push(current);
    }
    numbers.last().and_then(|n| n.parse::<f32>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use game::NlheConfig;

    #[test]
    fn encode_shape_matches_expected_input_dim() {
        let config = NlheConfig {
            num_players: 2,
            starting_stack: 2_000,
            small_blind: 10,
            big_blind: 20,
        };
        let game = NlheGame::new(config, 42).expect("game should initialize");
        let features = encode_nlhe_game(&game, 0, &[], &[]);
        assert_eq!(features.len(), INPUT_DIM);
    }

    #[test]
    fn action_token_parser_extracts_last_numeric_segment() {
        assert_eq!(extract_last_number("b:bet_0.33"), Some(0.33));
        assert_eq!(extract_last_number("r:raise_2.50"), Some(2.50));
        assert_eq!(extract_last_number("b:open_2.5bb"), Some(2.5));
        assert_eq!(extract_last_number("f"), None);
    }

    #[test]
    fn actor_aware_history_changes_feature_vector() {
        let config = NlheConfig {
            num_players: 6,
            starting_stack: 2_000,
            small_blind: 10,
            big_blind: 20,
        };
        let game = NlheGame::new(config, 42).expect("game should initialize");
        let tokens = vec!["c".to_string(), "r:3bet_2.50x".to_string()];
        let actor_hist_a = vec![0u8, 1u8];
        let actor_hist_b = vec![2u8, 3u8];
        let features_a = encode_nlhe_game(&game, 0, &tokens, &actor_hist_a);
        let features_b = encode_nlhe_game(&game, 0, &tokens, &actor_hist_b);
        assert_ne!(
            features_a, features_b,
            "actor-aware history should alter encoding"
        );
    }
}
