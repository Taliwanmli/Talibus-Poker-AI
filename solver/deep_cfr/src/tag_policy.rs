use cfr::nlhe_game::{enumerate_action_space, NlheState};
use game::{BettingRound, Card, LegalAction, NlheGame, Rank, Suit};
use rs_poker::core::{
    Card as RsCard, FlatHand, Rank as RsRank, Rankable, Suit as RsSuit, Value as RsValue,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const BLUFF_FREQUENCY: u64 = 12;

#[derive(Clone, Default)]
pub struct TagPolicy;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PreflopTier {
    Tier1,
    Tier2,
    Tier3,
    Tier4,
    Tier5,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PostflopClass {
    Monster,
    Strong,
    Medium,
    Draw,
    Weak,
}

impl TagPolicy {
    pub fn new() -> Self {
        Self
    }

    pub fn get_strategy(
        &mut self,
        state: &NlheState,
        player_idx: usize,
    ) -> Result<Vec<f64>, String> {
        let game = state
            .game
            .as_ref()
            .ok_or_else(|| "tag policy requires resolved NLHE state".to_string())?;
        let actor = game
            .current_actor()
            .ok_or_else(|| "tag policy requires active actor".to_string())?;
        if actor != player_idx {
            return Err(format!(
                "tag policy queried for player {player_idx} while actor is {actor}"
            ));
        }

        let action_space = enumerate_action_space(game, player_idx);
        if action_space.is_empty() {
            return Ok(Vec::new());
        }

        let to_call = amount_to_call(game, player_idx);
        let pot = pot_before_action(game).max(1);
        let spr = game.players[player_idx].stack as f64 / pot as f64;
        let bluff_allowed = should_bluff(state_hash(game, state, player_idx));

        let mut weights = vec![0.0; action_space.len()];
        match game.round {
            BettingRound::Preflop => {
                let tier = classify_preflop_tier(game.players[player_idx].hole_cards);
                apply_preflop_weights(&mut weights, &action_space, tier, to_call, pot, spr);
                Ok(normalize_or_uniform(weights))
            }
            BettingRound::Flop | BettingRound::Turn | BettingRound::River => {
                let class = classify_postflop_strength(game, player_idx);
                apply_postflop_weights(
                    &mut weights,
                    &action_space,
                    class,
                    to_call,
                    pot,
                    spr,
                    bluff_allowed,
                );
                Ok(normalize_or_uniform(weights))
            }
            BettingRound::Complete => {
                return Err("tag policy queried for completed game".to_string());
            }
        }
    }
}

fn apply_preflop_weights(
    weights: &mut [f64],
    action_space: &[cfr::nlhe_game::IndexedAction],
    tier: PreflopTier,
    to_call: u32,
    pot: u32,
    spr: f64,
) {
    let call_pressure = call_pressure(to_call, pot);
    for (idx, action) in action_space.iter().enumerate() {
        let ratio = amount_ratio(action.sort_amount, pot);
        let weight = match action.sort_group {
            0 => match tier {
                PreflopTier::Tier1 => {
                    if to_call > 0 {
                        0.03
                    } else {
                        0.0
                    }
                }
                PreflopTier::Tier2 => {
                    if to_call > 0 {
                        0.08
                    } else {
                        0.0
                    }
                }
                PreflopTier::Tier3 => {
                    if call_pressure > 0.30 {
                        0.95
                    } else {
                        0.30
                    }
                }
                PreflopTier::Tier4 => {
                    if to_call > 0 {
                        1.35
                    } else {
                        0.10
                    }
                }
                PreflopTier::Tier5 => {
                    if to_call > 0 {
                        1.80
                    } else {
                        0.15
                    }
                }
            },
            1 => match tier {
                PreflopTier::Tier1 => 0.30,
                PreflopTier::Tier2 => 0.55,
                PreflopTier::Tier3 => 0.95,
                PreflopTier::Tier4 => 1.20,
                PreflopTier::Tier5 => 1.50,
            },
            2 => match tier {
                PreflopTier::Tier1 => 1.10,
                PreflopTier::Tier2 => 0.95,
                PreflopTier::Tier3 => {
                    if call_pressure <= 0.30 {
                        0.75
                    } else {
                        0.40
                    }
                }
                PreflopTier::Tier4 => {
                    if call_pressure <= 0.15 {
                        0.35
                    } else {
                        0.08
                    }
                }
                PreflopTier::Tier5 => {
                    if call_pressure <= 0.06 {
                        0.05
                    } else {
                        0.01
                    }
                }
            },
            3 => match tier {
                PreflopTier::Tier1 => 1.60 * big_size_boost(ratio),
                PreflopTier::Tier2 => 1.15 * big_size_boost(ratio),
                PreflopTier::Tier3 => 0.55 * small_size_boost(ratio),
                PreflopTier::Tier4 => 0.15 * small_size_boost(ratio),
                PreflopTier::Tier5 => 0.02 * small_size_boost(ratio),
            },
            4 => match tier {
                PreflopTier::Tier1 => 1.95 * big_size_boost(ratio),
                PreflopTier::Tier2 => 1.20 * big_size_boost(ratio),
                PreflopTier::Tier3 => 0.28 * small_size_boost(ratio),
                PreflopTier::Tier4 => 0.07 * small_size_boost(ratio),
                PreflopTier::Tier5 => 0.01,
            },
            5 => match tier {
                PreflopTier::Tier1 => {
                    if spr < 2.0 {
                        0.95
                    } else {
                        0.30
                    }
                }
                PreflopTier::Tier2 => {
                    if spr < 1.6 {
                        0.50
                    } else {
                        0.12
                    }
                }
                PreflopTier::Tier3 => {
                    if spr < 1.0 {
                        0.18
                    } else {
                        0.02
                    }
                }
                PreflopTier::Tier4 | PreflopTier::Tier5 => 0.0,
            },
            _ => 0.0,
        };
        weights[idx] = weight.max(0.0);
    }
}

fn apply_postflop_weights(
    weights: &mut [f64],
    action_space: &[cfr::nlhe_game::IndexedAction],
    class: PostflopClass,
    to_call: u32,
    pot: u32,
    spr: f64,
    bluff_allowed: bool,
) {
    let pressure = call_pressure(to_call, pot);
    for (idx, action) in action_space.iter().enumerate() {
        let ratio = amount_ratio(action.sort_amount, pot);
        let weight = match class {
            PostflopClass::Monster => match action.sort_group {
                0 => 0.01,
                1 => 0.20,
                2 => 0.55,
                3 => 1.30 * big_size_boost(ratio),
                4 => 1.70 * big_size_boost(ratio),
                5 => {
                    if spr < 2.0 {
                        0.85
                    } else {
                        0.25
                    }
                }
                _ => 0.0,
            },
            PostflopClass::Strong => match action.sort_group {
                0 => {
                    if to_call > 0 {
                        0.08
                    } else {
                        0.0
                    }
                }
                1 => 0.55,
                2 => 0.90,
                3 => 1.05 * big_size_boost(ratio),
                4 => 1.10 * big_size_boost(ratio),
                5 => {
                    if spr < 1.5 {
                        0.45
                    } else {
                        0.12
                    }
                }
                _ => 0.0,
            },
            PostflopClass::Medium => match action.sort_group {
                0 => {
                    if to_call > 0 {
                        if pressure > 0.35 {
                            1.05
                        } else {
                            0.35
                        }
                    } else {
                        0.0
                    }
                }
                1 => 1.35,
                2 => {
                    if pressure <= 0.33 {
                        0.92
                    } else {
                        0.26
                    }
                }
                3 => {
                    if to_call == 0 {
                        0.35 * small_size_boost(ratio)
                    } else {
                        0.08
                    }
                }
                4 => 0.05,
                5 => 0.0,
                _ => 0.0,
            },
            PostflopClass::Draw => match action.sort_group {
                0 => {
                    if to_call > 0 {
                        if pressure > 0.32 {
                            0.82
                        } else {
                            0.22
                        }
                    } else {
                        0.0
                    }
                }
                1 => 1.10,
                2 => {
                    if pressure <= 0.28 {
                        0.95
                    } else {
                        0.30
                    }
                }
                3 => {
                    if to_call == 0 {
                        0.42 * small_size_boost(ratio)
                    } else {
                        0.24 * small_size_boost(ratio)
                    }
                }
                4 => {
                    if pressure <= 0.25 {
                        0.20 * small_size_boost(ratio)
                    } else {
                        0.05
                    }
                }
                5 => {
                    if spr < 1.0 {
                        0.05
                    } else {
                        0.0
                    }
                }
                _ => 0.0,
            },
            PostflopClass::Weak => match action.sort_group {
                0 => {
                    if to_call > 0 {
                        1.75
                    } else {
                        0.01
                    }
                }
                1 => 1.55,
                2 => {
                    if pressure < 0.10 {
                        0.12
                    } else {
                        0.02
                    }
                }
                3 => {
                    if bluff_allowed && to_call == 0 {
                        0.45 * small_size_boost(ratio)
                    } else {
                        0.02
                    }
                }
                4 => {
                    if bluff_allowed && to_call > 0 && pressure < 0.15 {
                        0.08 * small_size_boost(ratio)
                    } else {
                        0.01
                    }
                }
                5 => 0.0,
                _ => 0.0,
            },
        };
        weights[idx] = weight.max(0.0);
    }
}

fn classify_postflop_strength(game: &NlheGame, player_idx: usize) -> PostflopClass {
    let player = &game.players[player_idx];
    let rank = evaluate_rank(player.hole_cards, &game.board);
    match rank {
        RsRank::StraightFlush(_) | RsRank::FourOfAKind(_) | RsRank::FullHouse(_) => {
            PostflopClass::Monster
        }
        RsRank::Flush(_) | RsRank::Straight(_) | RsRank::ThreeOfAKind(_) | RsRank::TwoPair(_) => {
            PostflopClass::Strong
        }
        RsRank::OnePair(_) => {
            if has_top_pair_or_overpair(player.hole_cards, &game.board) {
                PostflopClass::Medium
            } else if has_draw(player.hole_cards, &game.board) {
                PostflopClass::Draw
            } else {
                PostflopClass::Weak
            }
        }
        RsRank::HighCard(_) => {
            if has_draw(player.hole_cards, &game.board) {
                PostflopClass::Draw
            } else {
                PostflopClass::Weak
            }
        }
    }
}

fn evaluate_rank(hole: [Card; 2], board: &[Card]) -> RsRank {
    let mut cards = Vec::with_capacity(2 + board.len());
    cards.push(to_rs_card(hole[0]));
    cards.push(to_rs_card(hole[1]));
    cards.extend(board.iter().copied().map(to_rs_card));
    FlatHand::new_with_cards(cards).rank()
}

fn has_top_pair_or_overpair(hole: [Card; 2], board: &[Card]) -> bool {
    if board.is_empty() {
        return false;
    }
    let board_high = board
        .iter()
        .map(|card| rank_value(card.rank))
        .max()
        .unwrap_or(0);
    let h0 = rank_value(hole[0].rank);
    let h1 = rank_value(hole[1].rank);

    let overpair = h0 == h1 && h0 > board_high;
    let top_pair = h0 == board_high || h1 == board_high;
    overpair || top_pair
}

fn has_draw(hole: [Card; 2], board: &[Card]) -> bool {
    if board.len() >= 5 {
        return false;
    }
    has_flush_draw(hole, board) || has_straight_draw(hole, board)
}

fn has_flush_draw(hole: [Card; 2], board: &[Card]) -> bool {
    if board.len() < 3 || board.len() >= 5 {
        return false;
    }
    let mut suit_counts = [0u8; 4];
    for card in [hole[0], hole[1]].into_iter().chain(board.iter().copied()) {
        let idx = suit_index(card.suit);
        suit_counts[idx] += 1;
    }
    suit_counts.into_iter().any(|count| count >= 4)
}

fn has_straight_draw(hole: [Card; 2], board: &[Card]) -> bool {
    if board.len() < 3 || board.len() >= 5 {
        return false;
    }

    let mut present = [false; 15];
    for card in [hole[0], hole[1]].into_iter().chain(board.iter().copied()) {
        let value = rank_value(card.rank) as usize;
        present[value] = true;
        if value == 14 {
            present[1] = true;
        }
    }

    for start in 1..=10 {
        let mut hits = 0usize;
        for value in start..=start + 4 {
            if present[value] {
                hits += 1;
            }
        }
        if hits >= 4 {
            return true;
        }
    }
    false
}

fn classify_preflop_tier(hole: [Card; 2]) -> PreflopTier {
    let r0 = rank_value(hole[0].rank);
    let r1 = rank_value(hole[1].rank);
    let high = r0.max(r1);
    let low = r0.min(r1);
    let pair = r0 == r1;
    let suited = hole[0].suit == hole[1].suit;
    let gap = high.saturating_sub(low);

    if pair {
        return match high {
            12..=14 => PreflopTier::Tier1,
            10..=11 => PreflopTier::Tier2,
            5..=9 => PreflopTier::Tier3,
            _ => PreflopTier::Tier4,
        };
    }

    if high == 14 && low == 13 && suited {
        return PreflopTier::Tier1;
    }
    if (high == 14 && low == 13) || (high == 14 && low == 12 && suited) {
        return PreflopTier::Tier2;
    }

    let broadway_combo = high >= 11 && low >= 10;
    let strong_suited_connector = suited && gap <= 1 && high >= 9 && low >= 7;
    let strong_suited_ace = suited && high == 14 && low >= 10;
    if broadway_combo || strong_suited_connector || strong_suited_ace {
        return PreflopTier::Tier3;
    }

    let weak_ace = high == 14;
    let small_suited_connector = suited && gap <= 2 && high >= 6;
    if weak_ace || small_suited_connector {
        return PreflopTier::Tier4;
    }

    PreflopTier::Tier5
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

fn suit_index(suit: Suit) -> usize {
    match suit {
        Suit::Clubs => 0,
        Suit::Diamonds => 1,
        Suit::Hearts => 2,
        Suit::Spades => 3,
    }
}

fn amount_to_call(game: &NlheGame, player_idx: usize) -> u32 {
    let legal = game.legal_actions(player_idx).unwrap_or_default();
    legal
        .iter()
        .find_map(|action| match action {
            LegalAction::Call { amount } => Some(*amount),
            _ => None,
        })
        .unwrap_or(0)
}

fn pot_before_action(game: &NlheGame) -> u32 {
    game.players
        .iter()
        .map(|player| player.total_contribution)
        .sum()
}

fn state_hash(game: &NlheGame, state: &NlheState, player_idx: usize) -> u64 {
    let mut hasher = DefaultHasher::new();
    player_idx.hash(&mut hasher);
    round_code(game.round).hash(&mut hasher);
    game.board.hash(&mut hasher);
    game.players[player_idx].hole_cards.hash(&mut hasher);
    state.action_history.hash(&mut hasher);
    hasher.finish()
}

fn round_code(round: BettingRound) -> u8 {
    match round {
        BettingRound::Preflop => 0,
        BettingRound::Flop => 1,
        BettingRound::Turn => 2,
        BettingRound::River => 3,
        BettingRound::Complete => 4,
    }
}

fn should_bluff(seed: u64) -> bool {
    seed % 100 < BLUFF_FREQUENCY
}

fn call_pressure(to_call: u32, pot: u32) -> f64 {
    if to_call == 0 {
        0.0
    } else {
        to_call as f64 / (pot.saturating_add(to_call).max(1) as f64)
    }
}

fn amount_ratio(sort_amount: u32, pot: u32) -> f64 {
    if sort_amount == 0 {
        0.5
    } else {
        sort_amount as f64 / pot.max(1) as f64
    }
}

fn big_size_boost(ratio: f64) -> f64 {
    (0.70 + ratio.clamp(0.20, 2.50) * 0.35).max(0.20)
}

fn small_size_boost(ratio: f64) -> f64 {
    (1.35 - ratio.clamp(0.20, 2.50) * 0.45).max(0.20)
}

fn normalize_or_uniform(mut weights: Vec<f64>) -> Vec<f64> {
    if weights.is_empty() {
        return weights;
    }
    for value in &mut weights {
        if !value.is_finite() || *value < 0.0 {
            *value = 0.0;
        }
    }
    let sum: f64 = weights.iter().sum();
    if sum > 1e-12 {
        weights.into_iter().map(|value| value / sum).collect()
    } else {
        vec![1.0 / weights.len() as f64; weights.len()]
    }
}

fn to_rs_card(card: Card) -> RsCard {
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
        Suit::Clubs => RsSuit::Club,
        Suit::Diamonds => RsSuit::Diamond,
        Suit::Hearts => RsSuit::Heart,
        Suit::Spades => RsSuit::Spade,
    };
    RsCard::new(value, suit)
}
