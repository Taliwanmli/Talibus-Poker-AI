use rand::prelude::{Rng, SeedableRng, SliceRandom, StdRng};
use std::collections::HashMap;

const NUM_ACTIONS: usize = 2;

#[derive(Clone, Debug, Default)]
struct InfoSet {
    regret_sum: [f64; NUM_ACTIONS],
    strategy_sum: [f64; NUM_ACTIONS],
}

impl InfoSet {
    fn strategy_from_regrets(&self) -> [f64; NUM_ACTIONS] {
        let positive_regrets = [self.regret_sum[0].max(0.0), self.regret_sum[1].max(0.0)];
        let normalizer = positive_regrets[0] + positive_regrets[1];
        if normalizer > 0.0 {
            [
                positive_regrets[0] / normalizer,
                positive_regrets[1] / normalizer,
            ]
        } else {
            [0.5, 0.5]
        }
    }

    fn average_strategy(&self) -> [f64; NUM_ACTIONS] {
        let normalizer = self.strategy_sum[0] + self.strategy_sum[1];
        if normalizer > 0.0 {
            [
                self.strategy_sum[0] / normalizer,
                self.strategy_sum[1] / normalizer,
            ]
        } else {
            [0.5, 0.5]
        }
    }
}

#[derive(Debug, Default)]
pub struct KuhnTrainer {
    infosets: HashMap<String, InfoSet>,
}

impl KuhnTrainer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn train(&mut self, iterations: usize, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..iterations {
            for traverser in 0..=1 {
                let cards = sample_private_cards(&mut rng);
                self.external_sampling_cfr(cards, "", traverser, 1.0, &mut rng);
            }
        }
    }

    pub fn infoset_count(&self) -> usize {
        self.infosets.len()
    }

    pub fn average_strategy_snapshot(&self) -> HashMap<String, [f64; NUM_ACTIONS]> {
        self.infosets
            .iter()
            .map(|(key, info)| (key.clone(), info.average_strategy()))
            .collect()
    }

    pub fn expected_value_player0(&self) -> f64 {
        let mut total = 0.0;
        let mut deals = 0usize;
        for c0 in 0..=2u8 {
            for c1 in 0..=2u8 {
                if c0 == c1 {
                    continue;
                }
                total += self.evaluate_average_policy([c0, c1], "");
                deals += 1;
            }
        }
        total / deals as f64
    }

    fn external_sampling_cfr(
        &mut self,
        cards: [u8; 2],
        history: &str,
        traverser: usize,
        traverser_reach: f64,
        rng: &mut StdRng,
    ) -> f64 {
        if let Some(terminal_u0) = terminal_utility_player0(cards, history) {
            return if traverser == 0 {
                terminal_u0
            } else {
                -terminal_u0
            };
        }

        let player = acting_player(history);
        let actions = legal_actions(history);
        let infoset_key = infoset_key(cards[player], history);

        let strategy = {
            let info = self.infosets.entry(infoset_key.clone()).or_default();
            info.strategy_from_regrets()
        };

        if player == traverser {
            let mut action_utilities = [0.0; NUM_ACTIONS];
            let mut node_utility = 0.0;
            for action_idx in 0..NUM_ACTIONS {
                let mut next_history = String::with_capacity(history.len() + 1);
                next_history.push_str(history);
                next_history.push(actions[action_idx]);
                action_utilities[action_idx] = self.external_sampling_cfr(
                    cards,
                    &next_history,
                    traverser,
                    traverser_reach * strategy[action_idx],
                    rng,
                );
                node_utility += strategy[action_idx] * action_utilities[action_idx];
            }

            let info = self
                .infosets
                .get_mut(&infoset_key)
                .expect("infoset must exist before regret update");
            for action_idx in 0..NUM_ACTIONS {
                info.regret_sum[action_idx] += action_utilities[action_idx] - node_utility;
            }
            node_utility
        } else {
            let info = self
                .infosets
                .get_mut(&infoset_key)
                .expect("infoset must exist before average-strategy update");
            for action_idx in 0..NUM_ACTIONS {
                info.strategy_sum[action_idx] += traverser_reach * strategy[action_idx];
            }

            let sampled_action = sample_action(strategy, rng);
            let mut next_history = String::with_capacity(history.len() + 1);
            next_history.push_str(history);
            next_history.push(actions[sampled_action]);
            self.external_sampling_cfr(cards, &next_history, traverser, traverser_reach, rng)
        }
    }

    fn evaluate_average_policy(&self, cards: [u8; 2], history: &str) -> f64 {
        if let Some(terminal_u0) = terminal_utility_player0(cards, history) {
            return terminal_u0;
        }

        let player = acting_player(history);
        let key = infoset_key(cards[player], history);
        let strategy = self
            .infosets
            .get(&key)
            .map(InfoSet::average_strategy)
            .unwrap_or([0.5, 0.5]);
        let actions = legal_actions(history);

        let mut expected_utility = 0.0;
        for action_idx in 0..NUM_ACTIONS {
            let mut next_history = String::with_capacity(history.len() + 1);
            next_history.push_str(history);
            next_history.push(actions[action_idx]);
            expected_utility +=
                strategy[action_idx] * self.evaluate_average_policy(cards, &next_history);
        }
        expected_utility
    }
}

fn sample_private_cards(rng: &mut StdRng) -> [u8; 2] {
    let mut deck = [0u8, 1, 2];
    deck.shuffle(rng);
    [deck[0], deck[1]]
}

fn acting_player(history: &str) -> usize {
    history.len() % 2
}

fn infoset_key(private_card: u8, history: &str) -> String {
    format!("{private_card}:{history}")
}

fn legal_actions(history: &str) -> [char; NUM_ACTIONS] {
    match history {
        "" | "c" => ['c', 'b'],
        "b" | "cb" => ['f', 'c'],
        _ => panic!("invalid non-terminal history: {history}"),
    }
}

fn sample_action(strategy: [f64; NUM_ACTIONS], rng: &mut StdRng) -> usize {
    let draw: f64 = rng.gen();
    if draw < strategy[0] {
        0
    } else {
        1
    }
}

fn terminal_utility_player0(cards: [u8; 2], history: &str) -> Option<f64> {
    match history {
        "cc" => Some(showdown_utility_player0(cards, 1.0)),
        "bc" | "cbc" => Some(showdown_utility_player0(cards, 2.0)),
        "bf" => Some(1.0),
        "cbf" => Some(-1.0),
        _ => None,
    }
}

fn showdown_utility_player0(cards: [u8; 2], amount: f64) -> f64 {
    if cards[0] > cards[1] {
        amount
    } else {
        -amount
    }
}

#[cfg(test)]
mod tests {
    use super::KuhnTrainer;

    #[test]
    fn converges_near_known_kuhn_value() {
        let mut trainer = KuhnTrainer::new();
        trainer.train(400_000, 7);
        let ev = trainer.expected_value_player0();

        // Nash value for first player in Kuhn is -1/18 ~= -0.05556.
        let nash_value = -1.0 / 18.0;
        assert!(
            (ev - nash_value).abs() < 0.03,
            "ev={ev}, target={nash_value}"
        );
    }

    #[test]
    fn training_discovers_non_trivial_infosets() {
        let mut trainer = KuhnTrainer::new();
        trainer.train(25_000, 99);
        assert!(trainer.infoset_count() >= 12);
    }
}
