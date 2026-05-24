use cfr::nlhe_game::{enumerate_action_space, NlheState};
use game::{LegalAction, NlheGame};

#[derive(Clone, Default)]
pub struct CallingStationPolicy;

impl CallingStationPolicy {
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
            .ok_or_else(|| "calling-station policy requires resolved NLHE state".to_string())?;
        let actor = game
            .current_actor()
            .ok_or_else(|| "calling-station policy requires active actor".to_string())?;
        if actor != player_idx {
            return Err(format!(
                "calling-station policy queried for player {player_idx} while actor is {actor}"
            ));
        }

        let action_space = enumerate_action_space(game, player_idx);
        if action_space.is_empty() {
            return Ok(Vec::new());
        }

        let to_call = amount_to_call(game, player_idx);
        let pot = pot_before_action(game).max(1);
        let pressure = call_pressure(to_call, pot);
        let mut weights = vec![0.0f64; action_space.len()];

        for (idx, action) in action_space.iter().enumerate() {
            let size_ratio = amount_ratio(action.sort_amount, pot);
            let weight = match action.sort_group {
                // Fold: rare, but allowed under heavy pressure.
                0 => {
                    if to_call == 0 {
                        0.0
                    } else if pressure > 0.70 {
                        0.35
                    } else {
                        0.10
                    }
                }
                // Check: preferred when free.
                1 => 2.50,
                // Call: dominant action profile.
                2 => {
                    if to_call > 0 {
                        6.00
                    } else {
                        0.0
                    }
                }
                // Bet: occasional probing when checked to.
                3 => {
                    if to_call == 0 {
                        0.35 * small_size_boost(size_ratio)
                    } else {
                        0.05
                    }
                }
                // Raise: almost never.
                4 => 0.02,
                // All-in: only as an exceptional continuation.
                5 => {
                    if to_call > 0 && pressure < 0.15 {
                        0.05
                    } else {
                        0.0
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
