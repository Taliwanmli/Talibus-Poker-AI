use abstraction::action_abstraction::{map_real_bet_to_abstract, AbstractAction, Street};
use rand::prelude::{Rng, SeedableRng, StdRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io;
use std::path::Path;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BlueprintInfoSetKey {
    pub card_bucket: u16,
    pub board_bucket: u16,
    pub action_history_hash: u64,
}

impl BlueprintInfoSetKey {
    pub fn encode(&self) -> String {
        format!(
            "{}:{}:{}",
            self.card_bucket, self.board_bucket, self.action_history_hash
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlueprintPolicy {
    pub action_probabilities: Vec<f64>,
}

impl BlueprintPolicy {
    pub fn normalized(&self) -> Vec<f64> {
        if self.action_probabilities.is_empty() {
            return Vec::new();
        }
        let mut values = self
            .action_probabilities
            .iter()
            .map(|v| if *v >= 0.0 { *v } else { 0.0 })
            .collect::<Vec<_>>();
        let sum: f64 = values.iter().sum();
        if sum <= 0.0 {
            vec![1.0 / values.len() as f64; values.len()]
        } else {
            values.iter_mut().for_each(|v| *v /= sum);
            values
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BlueprintTable {
    pub policies: HashMap<String, BlueprintPolicy>,
}

impl BlueprintTable {
    pub fn save_to_file(&self, path: &Path) -> io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        bincode::serialize_into(&mut file, self)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err.to_string()))
    }

    pub fn load_from_file(path: &Path) -> io::Result<Self> {
        let mut file = std::fs::File::open(path)?;
        bincode::deserialize_from(&mut file)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err.to_string()))
    }

    pub fn get_policy_by_key(&self, key: &str) -> Option<&BlueprintPolicy> {
        self.policies.get(key)
    }

    pub fn get_policy(&self, key: &BlueprintInfoSetKey) -> Option<&BlueprintPolicy> {
        self.get_policy_by_key(&key.encode())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlayerMode {
    Sample,
    ArgMax,
}

#[derive(Clone, Debug)]
pub struct BlueprintPlayer {
    table: BlueprintTable,
    mode: PlayerMode,
}

impl BlueprintPlayer {
    pub fn new(table: BlueprintTable, mode: PlayerMode) -> Self {
        Self { table, mode }
    }

    pub fn choose_action_index(
        &self,
        infoset: &BlueprintInfoSetKey,
        legal_actions_count: usize,
        seed: u64,
    ) -> usize {
        self.choose_action_index_for_key(&infoset.encode(), legal_actions_count, seed)
    }

    pub fn choose_action_index_for_key(
        &self,
        infoset_key: &str,
        legal_actions_count: usize,
        seed: u64,
    ) -> usize {
        if legal_actions_count == 0 {
            return 0;
        }
        let fallback = vec![1.0 / legal_actions_count as f64; legal_actions_count];
        let mut probs = self
            .table
            .get_policy_by_key(infoset_key)
            .map(BlueprintPolicy::normalized)
            .unwrap_or(fallback);

        if probs.len() < legal_actions_count {
            probs.resize(legal_actions_count, 0.0);
            let sum: f64 = probs.iter().sum();
            if sum > 0.0 {
                probs.iter_mut().for_each(|v| *v /= sum);
            } else {
                probs = vec![1.0 / legal_actions_count as f64; legal_actions_count];
            }
        } else if probs.len() > legal_actions_count {
            probs.truncate(legal_actions_count);
            let sum: f64 = probs.iter().sum();
            if sum > 0.0 {
                probs.iter_mut().for_each(|v| *v /= sum);
            } else {
                probs = vec![1.0 / legal_actions_count as f64; legal_actions_count];
            }
        }

        match self.mode {
            PlayerMode::ArgMax => {
                let mut best_idx = 0usize;
                let mut best_value = f64::NEG_INFINITY;
                for (idx, value) in probs.iter().copied().enumerate() {
                    if value > best_value {
                        best_value = value;
                        best_idx = idx;
                    }
                }
                best_idx
            }
            PlayerMode::Sample => {
                let mut rng = StdRng::seed_from_u64(seed);
                sample_index(&probs, &mut rng)
            }
        }
    }
}

pub fn translate_real_bet_to_abstract(
    street: Street,
    amount: f64,
    pot_before_bet: f64,
) -> AbstractAction {
    map_real_bet_to_abstract(street, amount, pot_before_bet)
}

fn sample_index(probabilities: &[f64], rng: &mut StdRng) -> usize {
    let mut cumulative = 0.0;
    let roll: f64 = rng.gen();
    for (idx, p) in probabilities.iter().copied().enumerate() {
        cumulative += p;
        if roll <= cumulative {
            return idx;
        }
    }
    probabilities.len().saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use crate::{
        translate_real_bet_to_abstract, BlueprintInfoSetKey, BlueprintPlayer, BlueprintPolicy,
        BlueprintTable, PlayerMode,
    };
    use abstraction::action_abstraction::{AbstractAction, Street};
    use std::collections::HashMap;
    use std::env;
    use std::fs;

    fn test_key() -> BlueprintInfoSetKey {
        BlueprintInfoSetKey {
            card_bucket: 7,
            board_bucket: 19,
            action_history_hash: 123456,
        }
    }

    #[test]
    fn argmax_mode_returns_highest_probability_action() {
        let key = test_key();
        let mut policies = HashMap::new();
        policies.insert(
            key.encode(),
            BlueprintPolicy {
                action_probabilities: vec![0.2, 0.6, 0.2],
            },
        );
        let table = BlueprintTable { policies };
        let player = BlueprintPlayer::new(table, PlayerMode::ArgMax);
        assert_eq!(player.choose_action_index(&key, 3, 1), 1);
    }

    #[test]
    fn sampling_mode_is_seed_deterministic() {
        let key = test_key();
        let mut policies = HashMap::new();
        policies.insert(
            key.encode(),
            BlueprintPolicy {
                action_probabilities: vec![0.1, 0.2, 0.7],
            },
        );
        let table = BlueprintTable { policies };
        let player = BlueprintPlayer::new(table, PlayerMode::Sample);

        let a = player.choose_action_index(&key, 3, 42);
        let b = player.choose_action_index(&key, 3, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn unknown_infoset_falls_back_to_uniform_legal_policy() {
        let table = BlueprintTable {
            policies: HashMap::new(),
        };
        let player = BlueprintPlayer::new(table, PlayerMode::ArgMax);
        assert_eq!(player.choose_action_index(&test_key(), 4, 7), 0);
    }

    #[test]
    fn blueprint_table_roundtrip() {
        let key = test_key();
        let mut policies = HashMap::new();
        policies.insert(
            key.encode(),
            BlueprintPolicy {
                action_probabilities: vec![0.3, 0.7],
            },
        );
        let table = BlueprintTable { policies };

        let path = env::temp_dir().join("talibus_blueprint_table.bin");
        table.save_to_file(&path).expect("save should work");
        let loaded = BlueprintTable::load_from_file(&path).expect("load should work");
        fs::remove_file(path).ok();

        assert_eq!(loaded.policies.len(), 1);
        assert!(loaded.get_policy(&key).is_some());
    }

    #[test]
    fn real_bet_translation_uses_action_abstraction_grid() {
        let mapped = translate_real_bet_to_abstract(Street::Turn, 70.0, 100.0);
        assert_eq!(mapped, AbstractAction::BetPotFraction(0.75));
    }
}
