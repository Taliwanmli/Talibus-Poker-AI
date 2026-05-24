#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Street {
    Preflop,
    Flop,
    Turn,
    River,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AbstractAction {
    Fold,
    Check,
    Call,
    BetPotFraction(f64),
    RaiseMultiplier(f64),
    OpenRaiseBbMultiple(f64),
    FourBetMultiplier(f64),
    AllIn,
}

const PREFLOP_OPEN_RAISE_GRID: [f64; 4] = [2.0, 2.5, 3.0, 3.5];
const FLOP_BET_GRID: [f64; 4] = [0.33, 0.67, 1.0, 1.5];
const TURN_BET_GRID: [f64; 4] = [0.5, 0.75, 1.0, 1.5];
const RIVER_BET_GRID: [f64; 4] = [0.33, 0.75, 1.25, 2.0];
const FLOP_RAISE_MULTIPLIER_GRID: [f64; 5] = [2.0, 2.5, 3.0, 3.5, 4.5];
const TURN_RAISE_MULTIPLIER_GRID: [f64; 5] = [2.0, 2.5, 3.0, 3.75, 5.0];
const RIVER_RAISE_MULTIPLIER_GRID: [f64; 5] = [1.8, 2.3, 3.0, 4.0, 5.5];
// More than one preflop re-raise size helps avoid jam-heavy collapse.
const PREFLOP_RERAISE_MULTIPLIER_GRID: [f64; 5] = [2.0, 2.3, 2.7, 3.2, 4.0];

pub fn preflop_open_raise_grid() -> &'static [f64] {
    &PREFLOP_OPEN_RAISE_GRID
}

pub fn preflop_reraise_multiplier_grid() -> &'static [f64] {
    &PREFLOP_RERAISE_MULTIPLIER_GRID
}

pub fn street_bet_grid(street: Street) -> &'static [f64] {
    match street {
        Street::Preflop => &PREFLOP_OPEN_RAISE_GRID,
        Street::Flop => &FLOP_BET_GRID,
        Street::Turn => &TURN_BET_GRID,
        Street::River => &RIVER_BET_GRID,
    }
}

pub fn street_raise_multiplier_grid(street: Street) -> &'static [f64] {
    match street {
        Street::Flop => &FLOP_RAISE_MULTIPLIER_GRID,
        Street::Turn => &TURN_RAISE_MULTIPLIER_GRID,
        Street::River => &RIVER_RAISE_MULTIPLIER_GRID,
        Street::Preflop => &[],
    }
}

pub fn nearest_preflop_open_raise_size(bb_multiple: f64) -> f64 {
    nearest_bucket(bb_multiple, &PREFLOP_OPEN_RAISE_GRID)
}

pub fn nearest_postflop_bet_fraction(street: Street, pot_fraction: f64) -> f64 {
    let grid: &[f64] = match street {
        Street::Flop => &FLOP_BET_GRID,
        Street::Turn => &TURN_BET_GRID,
        Street::River => &RIVER_BET_GRID,
        Street::Preflop => &PREFLOP_OPEN_RAISE_GRID,
    };
    nearest_bucket(pot_fraction, grid)
}

pub fn nearest_postflop_raise_multiplier(street: Street, _raise_multiple: f64) -> Option<f64> {
    let grid = street_raise_multiplier_grid(street);
    if grid.is_empty() {
        None
    } else {
        Some(nearest_bucket(_raise_multiple, grid))
    }
}

pub fn map_real_bet_to_abstract(
    street: Street,
    amount: f64,
    pot_before_bet: f64,
) -> AbstractAction {
    if amount <= 0.0 {
        return AbstractAction::Check;
    }
    if pot_before_bet <= 0.0 {
        return AbstractAction::AllIn;
    }

    match street {
        Street::Preflop => {
            let bb_multiple = amount / pot_before_bet;
            AbstractAction::OpenRaiseBbMultiple(nearest_preflop_open_raise_size(bb_multiple))
        }
        Street::Flop | Street::Turn | Street::River => {
            let pot_fraction = amount / pot_before_bet;
            AbstractAction::BetPotFraction(nearest_postflop_bet_fraction(street, pot_fraction))
        }
    }
}

fn nearest_bucket(target: f64, buckets: &[f64]) -> f64 {
    let mut best = buckets[0];
    let mut best_dist = (target - best).abs();
    for &candidate in &buckets[1..] {
        let dist = (target - candidate).abs();
        if dist < best_dist {
            best = candidate;
            best_dist = dist;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::{
        map_real_bet_to_abstract, nearest_postflop_bet_fraction, nearest_preflop_open_raise_size,
        preflop_reraise_multiplier_grid, street_raise_multiplier_grid, AbstractAction, Street,
    };

    #[test]
    fn preflop_open_raise_mapping_uses_nearest_bucket() {
        assert_eq!(nearest_preflop_open_raise_size(2.6), 2.5);
        assert_eq!(nearest_preflop_open_raise_size(2.9), 3.0);
    }

    #[test]
    fn postflop_bet_mapping_uses_street_specific_grid() {
        assert_eq!(nearest_postflop_bet_fraction(Street::Flop, 0.58), 0.67);
        assert_eq!(nearest_postflop_bet_fraction(Street::Turn, 0.62), 0.5);
        assert_eq!(nearest_postflop_bet_fraction(Street::River, 0.82), 0.75);
    }

    #[test]
    fn raise_and_preflop_reraise_constants_match_plan() {
        assert_eq!(street_raise_multiplier_grid(Street::Flop), &[2.0, 2.5, 3.0, 3.5, 4.5]);
        assert_eq!(street_raise_multiplier_grid(Street::Turn), &[2.0, 2.5, 3.0, 3.75, 5.0]);
        assert_eq!(
            street_raise_multiplier_grid(Street::River),
            &[1.8, 2.3, 3.0, 4.0, 5.5]
        );
        assert_eq!(preflop_reraise_multiplier_grid(), &[2.0, 2.3, 2.7, 3.2, 4.0]);
    }

    #[test]
    fn real_bet_translates_to_abstract_bucket() {
        let mapped = map_real_bet_to_abstract(Street::Flop, 80.0, 120.0);
        assert_eq!(mapped, AbstractAction::BetPotFraction(0.67));
    }
}
