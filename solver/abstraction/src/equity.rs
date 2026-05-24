use rand::prelude::{IndexedRandom, SeedableRng, StdRng};
use rs_poker::core::{Card, FlatHand, Rank, Rankable, Suit, Value};
use std::collections::HashSet;

pub const EQUITY_DIMS: usize = 8;

pub fn compute_equity_with_seed(
    hole: [Card; 2],
    board: &[Card],
    n_samples: usize,
    seed: u64,
) -> f64 {
    let mut rng = StdRng::seed_from_u64(seed);
    compute_equity(hole, board, n_samples, &mut rng)
}

pub fn compute_equity_multiway_with_seed(
    hole: [Card; 2],
    board: &[Card],
    opponent_count: usize,
    n_samples: usize,
    seed: u64,
) -> f64 {
    let mut rng = StdRng::seed_from_u64(seed);
    compute_equity_multiway(hole, board, opponent_count, n_samples, &mut rng)
}

pub fn compute_board_texture_equity_with_seed(board: &[Card], n_samples: usize, seed: u64) -> f64 {
    let mut rng = StdRng::seed_from_u64(seed);
    compute_board_texture_equity(board, n_samples, &mut rng)
}

pub fn compute_board_texture_equity_multiway_with_seed(
    board: &[Card],
    opponent_count: usize,
    n_samples: usize,
    seed: u64,
) -> f64 {
    let mut rng = StdRng::seed_from_u64(seed);
    compute_board_texture_equity_multiway(board, opponent_count, n_samples, &mut rng)
}

pub fn compute_equity(hole: [Card; 2], board: &[Card], n_samples: usize, rng: &mut StdRng) -> f64 {
    compute_equity_multiway(hole, board, 1, n_samples, rng)
}

pub fn compute_equity_multiway(
    hole: [Card; 2],
    board: &[Card],
    opponent_count: usize,
    n_samples: usize,
    rng: &mut StdRng,
) -> f64 {
    if n_samples == 0 || board.len() > 5 {
        return 0.5;
    }
    if opponent_count == 0 {
        return 1.0;
    }
    let mut used = HashSet::new();
    let hero_0 = u8::from(hole[0]);
    let hero_1 = u8::from(hole[1]);
    if !used.insert(hero_0) || !used.insert(hero_1) {
        return 0.5;
    }
    for &card in board {
        if !used.insert(u8::from(card)) {
            return 0.5;
        }
    }
    let deck = build_remaining_deck(&used);
    let board_runout_cards = 5usize.saturating_sub(board.len());
    let cards_needed = opponent_count.saturating_mul(2) + board_runout_cards;
    if deck.len() < cards_needed {
        return 0.5;
    }

    let mut hero_points = 0.0;
    for _ in 0..n_samples {
        let sampled = deck
            .choose_multiple(rng, cards_needed)
            .copied()
            .collect::<Vec<_>>();
        if sampled.len() != cards_needed {
            continue;
        }
        let runout_start = opponent_count * 2;
        let runout = &sampled[runout_start..];
        let hero_rank = rank_seven(hole, board, runout);
        let mut winners = 1usize;
        let mut hero_best = true;
        for opp_idx in 0..opponent_count {
            let base = opp_idx * 2;
            let villain = [sampled[base], sampled[base + 1]];
            let villain_rank = rank_seven(villain, board, runout);
            if villain_rank > hero_rank {
                hero_best = false;
                break;
            }
            if villain_rank == hero_rank {
                winners += 1;
            }
        }
        if hero_best {
            hero_points += 1.0 / winners as f64;
        }
    }
    hero_points / n_samples as f64
}

pub fn compute_board_texture_equity(board: &[Card], n_samples: usize, rng: &mut StdRng) -> f64 {
    compute_board_texture_equity_multiway(board, 1, n_samples, rng)
}

pub fn compute_board_texture_equity_multiway(
    board: &[Card],
    opponent_count: usize,
    n_samples: usize,
    rng: &mut StdRng,
) -> f64 {
    if n_samples == 0 || board.len() > 5 {
        return 0.5;
    }
    if opponent_count == 0 {
        return 1.0;
    }
    let mut used = HashSet::new();
    for &card in board {
        if !used.insert(u8::from(card)) {
            return 0.5;
        }
    }
    let deck = build_remaining_deck(&used);
    let board_runout_cards = 5usize.saturating_sub(board.len());
    let cards_needed = (1 + opponent_count).saturating_mul(2) + board_runout_cards;
    if deck.len() < cards_needed {
        return 0.5;
    }

    let mut hero_points = 0.0;
    for _ in 0..n_samples {
        let sampled = deck
            .choose_multiple(rng, cards_needed)
            .copied()
            .collect::<Vec<_>>();
        if sampled.len() != cards_needed {
            continue;
        }
        let hero = [sampled[0], sampled[1]];
        let runout_start = (1 + opponent_count) * 2;
        let runout = &sampled[runout_start..];
        let hero_rank = rank_seven(hero, board, runout);
        let mut winners = 1usize;
        let mut hero_best = true;
        for opp_idx in 0..opponent_count {
            let base = 2 + opp_idx * 2;
            let villain = [sampled[base], sampled[base + 1]];
            let villain_rank = rank_seven(villain, board, runout);
            if villain_rank > hero_rank {
                hero_best = false;
                break;
            }
            if villain_rank == hero_rank {
                winners += 1;
            }
        }
        if hero_best {
            hero_points += 1.0 / winners as f64;
        }
    }
    hero_points / n_samples as f64
}

pub fn equity_to_histogram(equity: f64) -> Vec<f64> {
    let mut hist = vec![0.0; EQUITY_DIMS];
    if hist.is_empty() {
        return hist;
    }
    if EQUITY_DIMS == 1 {
        hist[0] = 1.0;
        return hist;
    }

    let clamped = equity.clamp(0.0, 1.0);
    let position = clamped * EQUITY_DIMS as f64;
    let lower = (position.floor() as usize).min(EQUITY_DIMS - 1);
    let fraction = position - lower as f64;
    if lower + 1 < EQUITY_DIMS {
        hist[lower] = 1.0 - fraction;
        hist[lower + 1] = fraction;
    } else {
        hist[lower] = 1.0;
    }
    hist
}

fn rank_seven(hole: [Card; 2], board: &[Card], runout: &[Card]) -> Rank {
    let mut cards = Vec::with_capacity(7);
    cards.push(hole[0]);
    cards.push(hole[1]);
    cards.extend_from_slice(board);
    cards.extend_from_slice(runout);
    FlatHand::new_with_cards(cards).rank()
}

fn build_remaining_deck(used: &HashSet<u8>) -> Vec<Card> {
    let mut cards = Vec::with_capacity(52usize.saturating_sub(used.len()));
    for suit in Suit::suits() {
        for value in Value::values() {
            let card = Card::new(value, suit);
            if !used.contains(&u8::from(card)) {
                cards.push(card);
            }
        }
    }
    cards
}

#[cfg(test)]
mod tests {
    use super::{
        compute_board_texture_equity_multiway_with_seed, compute_board_texture_equity_with_seed,
        compute_equity_multiway_with_seed, compute_equity_with_seed, equity_to_histogram, EQUITY_DIMS,
    };
    use rs_poker::core::{Card, Suit, Value};

    #[test]
    fn aa_vs_kk_preflop_equity_is_reasonable() {
        let hole = [
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::Ace, Suit::Diamond),
        ];
        let equity = compute_equity_with_seed(hole, &[], 20_000, 17);
        assert!(equity > 0.75, "equity too low: {equity}");
        assert!(equity < 0.90, "equity too high: {equity}");
    }

    #[test]
    fn board_texture_equity_is_bounded() {
        let board = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Queen, Suit::Spade),
        ];
        let eq = compute_board_texture_equity_with_seed(&board, 10_000, 11);
        assert!((0.0..=1.0).contains(&eq), "equity out of range: {eq}");
    }

    #[test]
    fn multiway_equity_decreases_with_more_opponents() {
        let hole = [
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::King, Suit::Heart),
        ];
        let hu = compute_equity_multiway_with_seed(hole, &[], 1, 15_000, 91);
        let five_way = compute_equity_multiway_with_seed(hole, &[], 5, 15_000, 91);
        assert!(
            five_way < hu,
            "hero equity should decrease with more opponents: hu={hu}, sixmax={five_way}"
        );
    }

    #[test]
    fn board_texture_multiway_equity_is_bounded() {
        let board = [
            Card::new(Value::Ten, Suit::Heart),
            Card::new(Value::Nine, Suit::Heart),
            Card::new(Value::Two, Suit::Club),
        ];
        let eq = compute_board_texture_equity_multiway_with_seed(&board, 5, 5_000, 77);
        assert!((0.0..=1.0).contains(&eq), "multiway board equity out of range: {eq}");
    }

    #[test]
    fn equity_histogram_is_interpolated_and_normalized() {
        let hist = equity_to_histogram(0.42);
        assert_eq!(hist.len(), EQUITY_DIMS);
        let sum: f64 = hist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        assert!(
            (hist[3] - 0.64).abs() < 1e-9,
            "bin 3 weight mismatch: {}",
            hist[3]
        );
        assert!(
            (hist[4] - 0.36).abs() < 1e-9,
            "bin 4 weight mismatch: {}",
            hist[4]
        );
        assert!(
            hist.iter()
                .enumerate()
                .all(|(idx, value)| idx == 3 || idx == 4 || value.abs() < 1e-12),
            "unexpected non-zero bins: {:?}",
            hist
        );

        let hi = equity_to_histogram(1.0);
        assert!((hi[EQUITY_DIMS - 1] - 1.0).abs() < 1e-9);
    }
}
