pub mod action_abstraction;
pub mod clustering;
pub mod equity;

pub use equity::{
    compute_board_texture_equity, compute_board_texture_equity_with_seed, compute_equity,
    compute_equity_with_seed, equity_to_histogram, EQUITY_DIMS,
};

use rand::prelude::{IndexedRandom, SeedableRng, StdRng};
use rayon::prelude::*;
use rs_poker::core::{Card, FlatHand, Rankable, Suit, Value};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EquityError {
    InvalidCardEncoding,
    InvalidHandLength,
    InvalidBoardLength,
    DuplicateCard,
    NotEnoughCardsForRunout,
    NoIterations,
}

#[derive(Clone, Copy, Debug)]
pub struct EquityEstimate {
    pub hero_equity: f64,
    pub villain_equity: f64,
    pub tie_rate: f64,
}

pub fn estimate_heads_up_equity_from_str(
    hero: &str,
    villain: &str,
    board: &str,
    iterations: usize,
    seed: u64,
) -> Result<EquityEstimate, EquityError> {
    let hero_cards = parse_exact_hand(hero, 2)?;
    let villain_cards = parse_exact_hand(villain, 2)?;
    let board_cards = parse_variable_board(board)?;
    estimate_heads_up_equity(hero_cards, villain_cards, &board_cards, iterations, seed)
}

pub fn estimate_heads_up_equity(
    hero: [Card; 2],
    villain: [Card; 2],
    board: &[Card],
    iterations: usize,
    seed: u64,
) -> Result<EquityEstimate, EquityError> {
    if iterations == 0 {
        return Err(EquityError::NoIterations);
    }
    if board.len() > 5 {
        return Err(EquityError::InvalidBoardLength);
    }

    let mut used = HashSet::new();
    for card in hero
        .iter()
        .chain(villain.iter())
        .chain(board.iter())
        .copied()
    {
        let key = u8::from(card);
        if !used.insert(key) {
            return Err(EquityError::DuplicateCard);
        }
    }

    let deck = build_remaining_deck(&used);
    let cards_to_draw = 5usize.saturating_sub(board.len());
    if deck.len() < cards_to_draw {
        return Err(EquityError::NotEnoughCardsForRunout);
    }

    let (hero_points, tie_points) = (0..iterations)
        .into_par_iter()
        .map(|i| {
            let mut rng =
                StdRng::seed_from_u64(seed ^ ((i as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15)));
            let sampled = deck
                .choose_multiple(&mut rng, cards_to_draw)
                .copied()
                .collect::<Vec<_>>();

            let hero_rank = rank_seven(hero, board, &sampled);
            let villain_rank = rank_seven(villain, board, &sampled);
            if hero_rank > villain_rank {
                (1.0, 0.0)
            } else if hero_rank == villain_rank {
                (0.5, 1.0)
            } else {
                (0.0, 0.0)
            }
        })
        .reduce(
            || (0.0, 0.0),
            |left, right| (left.0 + right.0, left.1 + right.1),
        );

    let hero_equity = hero_points / iterations as f64;
    let tie_rate = tie_points / iterations as f64;
    Ok(EquityEstimate {
        hero_equity,
        villain_equity: 1.0 - hero_equity,
        tie_rate,
    })
}

fn rank_seven(hole: [Card; 2], board: &[Card], runout: &[Card]) -> rs_poker::core::Rank {
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

fn parse_exact_hand(raw: &str, expected_cards: usize) -> Result<[Card; 2], EquityError> {
    if expected_cards != 2 {
        return Err(EquityError::InvalidHandLength);
    }
    let hand = FlatHand::new_from_str(raw).map_err(|_| EquityError::InvalidCardEncoding)?;
    if hand.len() != expected_cards {
        return Err(EquityError::InvalidHandLength);
    }
    Ok([hand[0], hand[1]])
}

fn parse_variable_board(raw: &str) -> Result<Vec<Card>, EquityError> {
    let board = FlatHand::new_from_str(raw).map_err(|_| EquityError::InvalidCardEncoding)?;
    if board.len() > 5 {
        return Err(EquityError::InvalidBoardLength);
    }
    Ok(board.iter().copied().collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use crate::{estimate_heads_up_equity_from_str, EquityError};

    #[test]
    fn aa_vs_kk_preflop_equity_is_reasonable() {
        let estimate = estimate_heads_up_equity_from_str("AhAd", "KsKh", "", 50_000, 17)
            .expect("valid equity estimate");

        // Known value is ~81%. Keep a tolerance for Monte Carlo variance.
        assert!(
            estimate.hero_equity > 0.79,
            "equity too low: {}",
            estimate.hero_equity
        );
        assert!(
            estimate.hero_equity < 0.83,
            "equity too high: {}",
            estimate.hero_equity
        );
    }

    #[test]
    fn duplicate_cards_are_rejected() {
        let error = estimate_heads_up_equity_from_str("AhAd", "AhKs", "", 10_000, 1)
            .expect_err("must fail when duplicate cards are present");
        assert_eq!(error, EquityError::DuplicateCard);
    }
}
