use cfr::nlhe_game::{enumerate_action_space, NlheState};
use game::{BettingRound, LegalAction, NlheGame};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const LAG_BLUFF_FREQUENCY: u64 = 28;

#[derive(Clone, Default)]
pub struct LagPolicy;

impl LagPolicy {
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
            .ok_or_else(|| "lag policy requires resolved NLHE state".to_string())?;
        let actor = game
            .current_actor()
            .ok_or_else(|| "lag policy requires active actor".to_string())?;
        if actor != player_idx {
            return Err(format!(
                "lag policy queried for player {player_idx} while actor is {actor}"
            ));
        }

        let action_space = enumerate_action_space(game, player_idx);
        if action_space.is_empty() {
            return Ok(Vec::new());
        }

        let to_call = amount_to_call(game, player_idx);
        let pot = pot_before_action(game).max(1);
        let pressure = call_pressure(to_call, pot);
        let spr = game.players[player_idx].stack as f64 / pot as f64;
        let bluff_allowed = should_bluff(state_hash(game, state, player_idx));
        let is_preflop = game.round == BettingRound::Preflop;

        let mut weights = vec![0.0f64; action_space.len()];
        for (idx, action) in action_space.iter().enumerate() {
            let ratio = amount_ratio(action.sort_amount, pot);
            let weight = match action.sort_group {
                // Fold rarely under pressure.
                0 => {
                    if to_call == 0 {
                        0.0
                    } else if pressure > 0.55 {
                        0.35
                    } else {
                        0.08
                    }
                }
                // Check less often than passive profiles.
                1 => {
                    if to_call == 0 {
                        if is_preflop {
                            0.35
                        } else {
                            0.25
                        }
                    } else {
                        0.0
                    }
                }
                // Calls still exist but secondary to aggression.
                2 => {
                    if to_call == 0 {
                        0.0
                    } else if pressure < 0.35 {
                        1.25
                    } else {
                        0.70
                    }
                }
                // Frequent bets, with bias toward practical sizing.
                3 => {
                    if to_call == 0 {
                        if bluff_allowed {
                            2.20 * small_size_boost(ratio)
                        } else {
                            1.60 * big_size_boost(ratio)
                        }
                    } else if bluff_allowed {
                        0.55 * small_size_boost(ratio)
                    } else {
                        0.22
                    }
                }
                // Raises are a core LAG behavior.
                4 => {
                    if to_call > 0 {
                        if bluff_allowed {
                            2.60 * small_size_boost(ratio)
                        } else {
                            2.10 * big_size_boost(ratio)
                        }
                    } else {
                        1.40 * big_size_boost(ratio)
                    }
                }
                // All-in still bounded to realistic low-SPR spots.
                5 => {
                    if spr < 1.20 {
                        0.45
                    } else if bluff_allowed && pressure < 0.20 {
                        0.10
                    } else {
                        0.03
                    }
                }
                _ => 0.0,
            };
            weights[idx] = weight.max(0.0);
        }

        Ok(normalize_or_uniform(weights))
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

fn call_pressure(to_call: u32, pot: u32) -> f64 {
    if to_call == 0 {
        0.0
    } else {
        to_call as f64 / pot.saturating_add(to_call).max(1) as f64
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
    seed % 100 < LAG_BLUFF_FREQUENCY
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
