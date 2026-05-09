use dashmap::DashMap;
use rand::prelude::{Rng, SeedableRng, StdRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NodeKind {
    Terminal,
    Chance,
    Player(usize),
}

pub trait GameModel {
    type State: Clone + Send + Sync;

    fn root_state(&self) -> Self::State;
    fn node_kind(&self, state: &Self::State) -> NodeKind;
    fn legal_action_count(&self, state: &Self::State) -> usize;
    fn next_state(&self, state: &Self::State, action_idx: usize) -> Self::State;
    fn terminal_utility(&self, state: &Self::State, player: usize) -> f64;
    fn infoset_key(&self, state: &Self::State, player: usize) -> String;
    fn chance_probabilities(&self, state: &Self::State) -> Vec<f64>;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InfoSetNode {
    regret_sum: Vec<f64>,
    strategy_sum: Vec<f64>,
}

impl InfoSetNode {
    fn new(action_count: usize) -> Self {
        Self {
            regret_sum: vec![0.0; action_count],
            strategy_sum: vec![0.0; action_count],
        }
    }

    fn ensure_action_count(&mut self, action_count: usize) {
        if self.regret_sum.len() < action_count {
            self.regret_sum.resize(action_count, 0.0);
        }
        if self.strategy_sum.len() < action_count {
            self.strategy_sum.resize(action_count, 0.0);
        }
    }

    fn strategy_from_regrets(&self, action_count: usize) -> Vec<f64> {
        let mut positive = vec![0.0; action_count];
        let mut normalizer = 0.0;
        for (idx, slot) in positive.iter_mut().enumerate().take(action_count) {
            let value = self.regret_sum[idx].max(0.0);
            *slot = value;
            normalizer += value;
        }
        if normalizer > 0.0 {
            positive.iter_mut().for_each(|p| *p /= normalizer);
            positive
        } else {
            vec![1.0 / action_count as f64; action_count]
        }
    }

    fn average_strategy(&self, action_count: usize) -> Vec<f64> {
        let normalizer: f64 = self.strategy_sum.iter().take(action_count).sum();
        if normalizer > 0.0 {
            self.strategy_sum
                .iter()
                .take(action_count)
                .map(|value| value / normalizer)
                .collect()
        } else {
            vec![1.0 / action_count as f64; action_count]
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExternalSamplingTrainer {
    pub num_players: usize,
    pub iterations: u64,
    table: DashMap<String, InfoSetNode>,
}

#[derive(Clone, Debug)]
pub struct ExploitabilityReport {
    pub expected_values: Vec<f64>,
    pub best_response_values: Vec<f64>,
    pub exploitability: f64,
}

impl ExternalSamplingTrainer {
    pub fn new(num_players: usize) -> Self {
        Self {
            num_players,
            iterations: 0,
            table: DashMap::new(),
        }
    }

    pub fn infoset_count(&self) -> usize {
        self.table.len()
    }

    // #region agent log
    pub fn diagnostic_info(&self) -> (usize, usize, f64, usize) {
        let total = self.table.len();
        let mut non_uniform = 0usize;
        let mut total_actions = 0usize;
        let mut max_actions = 0usize;
        for entry in self.table.iter() {
            let node = entry.value();
            let ac = node.regret_sum.len();
            total_actions += ac;
            if ac > max_actions {
                max_actions = ac;
            }
            let strategy = node.average_strategy(ac);
            let uniform = 1.0 / ac.max(1) as f64;
            if strategy.iter().any(|p| (p - uniform).abs() > 0.01) {
                non_uniform += 1;
            }
        }
        let avg_actions = if total > 0 {
            total_actions as f64 / total as f64
        } else {
            0.0
        };
        (total, non_uniform, avg_actions, max_actions)
    }

    pub fn sample_infosets_debug(
        &self,
        count: usize,
    ) -> Vec<(String, usize, Vec<f64>, Vec<f64>, Vec<f64>)> {
        self.table
            .iter()
            .take(count)
            .map(|entry| {
                let key = entry.key().clone();
                let node = entry.value();
                let ac = node.regret_sum.len();
                (
                    key,
                    ac,
                    node.average_strategy(ac),
                    node.regret_sum.clone(),
                    node.strategy_sum.clone(),
                )
            })
            .collect()
    }
    // #endregion

    pub fn average_policy_table(&self) -> HashMap<String, Vec<f64>> {
        let mut out = HashMap::with_capacity(self.table.len());
        for entry in self.table.iter() {
            let key = entry.key();
            let node = entry.value();
            let action_count = node.regret_sum.len().max(node.strategy_sum.len());
            if action_count == 0 {
                continue;
            }
            out.insert(key.clone(), node.average_strategy(action_count));
        }
        out
    }

    pub fn train_serial<G: GameModel>(&mut self, game: &G, iterations: usize, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..iterations {
            self.iterations += 1;
            let weight = (self.iterations.max(1)) as f64;
            for traverser in 0..self.num_players {
                let root = game.root_state();
                traverse_table(&self.table, game, root, traverser, weight, 1.0, &mut rng);
            }
        }
    }

    /// Parallel training using a shared DashMap table.
    pub fn train_parallel<G: GameModel + Sync>(
        &mut self,
        game: &G,
        iterations: usize,
        workers: usize,
        seed: u64,
    ) {
        if iterations == 0 {
            return;
        }
        if workers <= 1 {
            self.train_serial(game, iterations, seed);
            return;
        }

        let iterations_before = self.iterations;
        let num_players = self.num_players;
        let table = &self.table;

        let chunks = split_iterations(iterations, workers);
        let mut running_offset = 0u64;
        let offsets: Vec<u64> = chunks
            .iter()
            .map(|chunk| {
                let o = running_offset;
                running_offset += *chunk as u64;
                o
            })
            .collect();

        chunks
            .into_par_iter()
            .enumerate()
            .for_each(|(worker_idx, chunk)| {
                if chunk == 0 {
                    return;
                }
                let start_weight = iterations_before + offsets[worker_idx] + 1;
                let worker_seed =
                    seed ^ ((worker_idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let mut rng = StdRng::seed_from_u64(worker_seed);
                for t in 0..chunk {
                    let weight = (start_weight + t as u64) as f64;
                    for traverser in 0..num_players {
                        let root = game.root_state();
                        traverse_table(table, game, root, traverser, weight, 1.0, &mut rng);
                    }
                }
            });

        self.iterations += iterations as u64;
    }

    pub fn expected_value_against_average_policy<G: GameModel>(
        &self,
        game: &G,
        player: usize,
    ) -> f64 {
        self.expected_utility_average(game, game.root_state(), player)
    }

    /// Measures **oracle exploitability** for two-player games using a state-wise
    /// best response. This is an upper-bound style diagnostic and can overestimate
    /// true exploitability in imperfect-information games because the response is
    /// selected at underlying states (not constrained to infosets).
    pub fn measure_oracle_exploitability_two_player<G: GameModel>(
        &self,
        game: &G,
    ) -> Option<ExploitabilityReport> {
        if self.num_players != 2 {
            return None;
        }
        let mut expected_values = Vec::with_capacity(self.num_players);
        let mut br_values = Vec::with_capacity(self.num_players);

        for player in 0..self.num_players {
            expected_values.push(self.expected_utility_average(game, game.root_state(), player));
            br_values.push(self.best_response_value(game, game.root_state(), player));
        }

        let mut total_advantage = 0.0;
        for idx in 0..self.num_players {
            total_advantage += br_values[idx] - expected_values[idx];
        }
        let exploitability = total_advantage / self.num_players as f64;

        Some(ExploitabilityReport {
            expected_values,
            best_response_values: br_values,
            exploitability,
        })
    }

    /// Backward-compatible alias. For imperfect-information games this is an
    /// oracle/state-wise exploitability proxy, not exact Nash exploitability.
    pub fn measure_exploitability_two_player<G: GameModel>(
        &self,
        game: &G,
    ) -> Option<ExploitabilityReport> {
        self.measure_oracle_exploitability_two_player(game)
    }

    fn infoset_br_action_values<G: GameModel>(
        &self,
        game: &G,
        state: G::State,
        br_player: usize,
        opponent_reach: f64,
        infoset_action_values: &mut HashMap<String, Vec<f64>>,
    ) -> f64 {
        match game.node_kind(&state) {
            NodeKind::Terminal => game.terminal_utility(&state, br_player),
            NodeKind::Chance => {
                let probs = game.chance_probabilities(&state);
                probs
                    .iter()
                    .enumerate()
                    .map(|(idx, p)| {
                        p * self.infoset_br_action_values(
                            game,
                            game.next_state(&state, idx),
                            br_player,
                            opponent_reach * p,
                            infoset_action_values,
                        )
                    })
                    .sum()
            }
            NodeKind::Player(player_idx) => {
                let action_count = game.legal_action_count(&state);
                if action_count == 0 {
                    return game.terminal_utility(&state, br_player);
                }

                if player_idx == br_player {
                    let key = game.infoset_key(&state, player_idx);
                    let mut action_returns = vec![0.0; action_count];
                    for action_idx in 0..action_count {
                        action_returns[action_idx] = self.infoset_br_action_values(
                            game,
                            game.next_state(&state, action_idx),
                            br_player,
                            opponent_reach,
                            infoset_action_values,
                        );
                    }

                    {
                        let entry = infoset_action_values
                            .entry(key)
                            .or_insert_with(|| vec![0.0; action_count]);
                        if entry.len() < action_count {
                            entry.resize(action_count, 0.0);
                        }
                        for action_idx in 0..action_count {
                            entry[action_idx] += opponent_reach * action_returns[action_idx];
                        }
                    }

                    action_returns
                        .into_iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                        .unwrap_or(0.0)
                } else {
                    let key = game.infoset_key(&state, player_idx);
                    let strategy = self
                        .table
                        .get(&key)
                        .map(|node| node.average_strategy(action_count))
                        .unwrap_or_else(|| vec![1.0 / action_count as f64; action_count]);
                    strategy
                        .iter()
                        .enumerate()
                        .map(|(idx, prob)| {
                            prob * self.infoset_br_action_values(
                                game,
                                game.next_state(&state, idx),
                                br_player,
                                opponent_reach * prob,
                                infoset_action_values,
                            )
                        })
                        .sum()
                }
            }
        }
    }

    fn infoset_br_fixed_value<G: GameModel>(
        &self,
        game: &G,
        state: G::State,
        br_player: usize,
        infoset_best_actions: &HashMap<String, usize>,
    ) -> f64 {
        match game.node_kind(&state) {
            NodeKind::Terminal => game.terminal_utility(&state, br_player),
            NodeKind::Chance => {
                let probs = game.chance_probabilities(&state);
                probs
                    .iter()
                    .enumerate()
                    .map(|(idx, p)| {
                        p * self.infoset_br_fixed_value(
                            game,
                            game.next_state(&state, idx),
                            br_player,
                            infoset_best_actions,
                        )
                    })
                    .sum()
            }
            NodeKind::Player(player_idx) => {
                let action_count = game.legal_action_count(&state);
                if action_count == 0 {
                    return game.terminal_utility(&state, br_player);
                }

                if player_idx == br_player {
                    let key = game.infoset_key(&state, player_idx);
                    let action_idx = infoset_best_actions
                        .get(&key)
                        .copied()
                        .filter(|idx| *idx < action_count)
                        .unwrap_or(0);
                    self.infoset_br_fixed_value(
                        game,
                        game.next_state(&state, action_idx),
                        br_player,
                        infoset_best_actions,
                    )
                } else {
                    let key = game.infoset_key(&state, player_idx);
                    let strategy = self
                        .table
                        .get(&key)
                        .map(|node| node.average_strategy(action_count))
                        .unwrap_or_else(|| vec![1.0 / action_count as f64; action_count]);
                    strategy
                        .iter()
                        .enumerate()
                        .map(|(idx, prob)| {
                            prob * self.infoset_br_fixed_value(
                                game,
                                game.next_state(&state, idx),
                                br_player,
                                infoset_best_actions,
                            )
                        })
                        .sum()
                }
            }
        }
    }

    pub fn measure_infoset_exploitability_two_player<G: GameModel>(
        &self,
        game: &G,
    ) -> Option<ExploitabilityReport> {
        if self.num_players != 2 {
            return None;
        }

        let mut expected_values = Vec::with_capacity(self.num_players);
        let mut br_values = Vec::with_capacity(self.num_players);

        for player in 0..self.num_players {
            expected_values.push(self.expected_utility_average(game, game.root_state(), player));
        }

        for br_player in 0..self.num_players {
            let mut infoset_action_values = HashMap::<String, Vec<f64>>::new();
            self.infoset_br_action_values(
                game,
                game.root_state(),
                br_player,
                1.0,
                &mut infoset_action_values,
            );

            let mut infoset_best_actions = HashMap::<String, usize>::new();
            for (key, values) in infoset_action_values {
                if values.is_empty() {
                    continue;
                }
                let best_action = values
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                infoset_best_actions.insert(key, best_action);
            }

            br_values.push(self.infoset_br_fixed_value(
                game,
                game.root_state(),
                br_player,
                &infoset_best_actions,
            ));
        }

        let mut total_advantage = 0.0;
        for idx in 0..self.num_players {
            total_advantage += br_values[idx] - expected_values[idx];
        }
        let exploitability = total_advantage / self.num_players as f64;

        Some(ExploitabilityReport {
            expected_values,
            best_response_values: br_values,
            exploitability,
        })
    }

    /// Monte Carlo approximation of two-player infoset exploitability.
    ///
    /// Instead of expanding all root chance outcomes, this samples
    /// `num_samples` root deals and computes a fixed infoset best response
    /// against that sample set.
    pub fn measure_sampled_exploitability_two_player<G: GameModel>(
        &self,
        game: &G,
        num_samples: usize,
        seed: u64,
    ) -> Option<ExploitabilityReport> {
        if self.num_players != 2 {
            return None;
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let sample_count = num_samples.max(1);
        let root = game.root_state();
        let mut sampled_roots: Vec<(G::State, f64)> = Vec::with_capacity(sample_count);

        match game.node_kind(&root) {
            NodeKind::Chance => {
                let probs = game.chance_probabilities(&root);
                if probs.is_empty() {
                    sampled_roots.push((root, 1.0));
                } else if sample_count >= probs.len() {
                    for (action_idx, p) in probs.iter().enumerate() {
                        if *p > 0.0 {
                            sampled_roots.push((game.next_state(&root, action_idx), *p));
                        }
                    }
                } else {
                    for _ in 0..sample_count {
                        let action_idx = sample_from_probs(&probs, &mut rng);
                        sampled_roots.push((
                            game.next_state(&root, action_idx),
                            1.0 / sample_count as f64,
                        ));
                    }
                }
            }
            _ => {
                sampled_roots.push((root, 1.0));
            }
        }

        let mut expected_values = vec![0.0; self.num_players];
        for (state, weight) in &sampled_roots {
            for (player, slot) in expected_values.iter_mut().enumerate() {
                *slot += *weight * self.expected_utility_average(game, state.clone(), player);
            }
        }

        let mut br_values = Vec::with_capacity(self.num_players);
        for br_player in 0..self.num_players {
            let mut infoset_action_values = HashMap::<String, Vec<f64>>::new();
            for (state, weight) in &sampled_roots {
                self.infoset_br_action_values(
                    game,
                    state.clone(),
                    br_player,
                    *weight,
                    &mut infoset_action_values,
                );
            }

            let mut infoset_best_actions = HashMap::<String, usize>::new();
            for (key, values) in infoset_action_values {
                if values.is_empty() {
                    continue;
                }
                let best_action = values
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                infoset_best_actions.insert(key, best_action);
            }

            let mut br_total = 0.0;
            for (state, weight) in &sampled_roots {
                br_total += *weight
                    * self.infoset_br_fixed_value(
                        game,
                        state.clone(),
                        br_player,
                        &infoset_best_actions,
                    );
            }
            br_values.push(br_total);
        }

        let mut total_advantage = 0.0;
        for idx in 0..self.num_players {
            total_advantage += br_values[idx] - expected_values[idx];
        }
        let exploitability = total_advantage / self.num_players as f64;

        Some(ExploitabilityReport {
            expected_values,
            best_response_values: br_values,
            exploitability,
        })
    }

    pub fn save_checkpoint(&self, path: &Path) -> io::Result<()> {
        let mut file = File::create(path)?;
        bincode::serialize_into(&mut file, self)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err.to_string()))
    }

    pub fn load_checkpoint(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;
        bincode::deserialize_from(&mut file)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err.to_string()))
    }

    fn expected_utility_average<G: GameModel>(
        &self,
        game: &G,
        state: G::State,
        target_player: usize,
    ) -> f64 {
        match game.node_kind(&state) {
            NodeKind::Terminal => game.terminal_utility(&state, target_player),
            NodeKind::Chance => {
                let probs = game.chance_probabilities(&state);
                probs
                    .iter()
                    .enumerate()
                    .map(|(idx, p)| {
                        p * self.expected_utility_average(
                            game,
                            game.next_state(&state, idx),
                            target_player,
                        )
                    })
                    .sum()
            }
            NodeKind::Player(player_idx) => {
                let action_count = game.legal_action_count(&state);
                let key = game.infoset_key(&state, player_idx);
                let strategy = self
                    .table
                    .get(&key)
                    .map(|node| node.average_strategy(action_count))
                    .unwrap_or_else(|| vec![1.0 / action_count as f64; action_count]);
                strategy
                    .iter()
                    .enumerate()
                    .map(|(idx, prob)| {
                        prob * self.expected_utility_average(
                            game,
                            game.next_state(&state, idx),
                            target_player,
                        )
                    })
                    .sum()
            }
        }
    }

    fn best_response_value<G: GameModel>(
        &self,
        game: &G,
        state: G::State,
        br_player: usize,
    ) -> f64 {
        match game.node_kind(&state) {
            NodeKind::Terminal => game.terminal_utility(&state, br_player),
            NodeKind::Chance => {
                let probs = game.chance_probabilities(&state);
                probs
                    .iter()
                    .enumerate()
                    .map(|(idx, p)| {
                        p * self.best_response_value(game, game.next_state(&state, idx), br_player)
                    })
                    .sum()
            }
            NodeKind::Player(player_idx) => {
                let action_count = game.legal_action_count(&state);
                if player_idx == br_player {
                    (0..action_count)
                        .map(|idx| {
                            self.best_response_value(game, game.next_state(&state, idx), br_player)
                        })
                        .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                        .unwrap_or(0.0)
                } else {
                    let key = game.infoset_key(&state, player_idx);
                    let strategy = self
                        .table
                        .get(&key)
                        .map(|node| node.average_strategy(action_count))
                        .unwrap_or_else(|| vec![1.0 / action_count as f64; action_count]);
                    strategy
                        .iter()
                        .enumerate()
                        .map(|(idx, prob)| {
                            prob * self.best_response_value(
                                game,
                                game.next_state(&state, idx),
                                br_player,
                            )
                        })
                        .sum()
                }
            }
        }
    }
}

fn traverse_table<G: GameModel>(
    table: &DashMap<String, InfoSetNode>,
    game: &G,
    state: G::State,
    traverser: usize,
    iteration_weight: f64,
    traverser_reach: f64,
    rng: &mut StdRng,
) -> f64 {
    match game.node_kind(&state) {
        NodeKind::Terminal => game.terminal_utility(&state, traverser),
        NodeKind::Chance => {
            let probs = game.chance_probabilities(&state);
            let sampled = sample_from_probs(&probs, rng);
            let next = game.next_state(&state, sampled);
            traverse_table(
                table,
                game,
                next,
                traverser,
                iteration_weight,
                traverser_reach,
                rng,
            )
        }
        NodeKind::Player(player_idx) => {
            let action_count = game.legal_action_count(&state);
            if action_count == 0 {
                return game.terminal_utility(&state, traverser);
            }
            let key = game.infoset_key(&state, player_idx);
            let strategy = {
                let mut node = table
                    .entry(key.clone())
                    .or_insert_with(|| InfoSetNode::new(action_count));
                node.ensure_action_count(action_count);
                node.strategy_from_regrets(action_count)
            };

            if player_idx == traverser {
                let mut action_values = vec![0.0; action_count];
                let mut node_value = 0.0;
                for action_idx in 0..action_count {
                    let next = game.next_state(&state, action_idx);
                    action_values[action_idx] = traverse_table(
                        table,
                        game,
                        next,
                        traverser,
                        iteration_weight,
                        traverser_reach * strategy[action_idx],
                        rng,
                    );
                    node_value += strategy[action_idx] * action_values[action_idx];
                }

                let mut node = table
                    .get_mut(&key)
                    .expect("infoset node must exist before update");
                for action_idx in 0..action_count {
                    node.regret_sum[action_idx] += action_values[action_idx] - node_value;
                    node.strategy_sum[action_idx] +=
                        iteration_weight * traverser_reach * strategy[action_idx];
                }
                node_value
            } else {
                {
                    let mut node = table
                        .get_mut(&key)
                        .expect("infoset node must exist before sampling");
                    for action_idx in 0..action_count {
                        node.strategy_sum[action_idx] +=
                            iteration_weight * traverser_reach * strategy[action_idx];
                    }
                }

                let sampled_action = sample_from_probs(&strategy, rng);
                let next = game.next_state(&state, sampled_action);
                traverse_table(
                    table,
                    game,
                    next,
                    traverser,
                    iteration_weight,
                    traverser_reach,
                    rng,
                )
            }
        }
    }
}

fn sample_from_probs(probabilities: &[f64], rng: &mut StdRng) -> usize {
    if probabilities.is_empty() {
        return 0;
    }
    let mut cumulative = 0.0;
    let draw: f64 = rng.gen();
    for (idx, prob) in probabilities.iter().enumerate() {
        cumulative += *prob;
        if draw <= cumulative {
            return idx;
        }
    }
    probabilities.len() - 1
}

fn split_iterations(iterations: usize, workers: usize) -> Vec<usize> {
    let base = iterations / workers;
    let extra = iterations % workers;
    (0..workers)
        .map(|idx| if idx < extra { base + 1 } else { base })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{ExternalSamplingTrainer, GameModel, NodeKind};
    use std::env;
    use std::fs;

    #[derive(Clone)]
    struct KuhnState {
        cards: Option<[u8; 2]>,
        history: String,
    }

    struct KuhnGame;

    impl KuhnGame {
        const DEALS: [[u8; 2]; 6] = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]];
    }

    impl GameModel for KuhnGame {
        type State = KuhnState;

        fn root_state(&self) -> Self::State {
            KuhnState {
                cards: None,
                history: String::new(),
            }
        }

        fn node_kind(&self, state: &Self::State) -> NodeKind {
            if state.cards.is_none() {
                return NodeKind::Chance;
            }
            if terminal_utility_player0(state.cards.expect("cards"), &state.history).is_some() {
                NodeKind::Terminal
            } else {
                NodeKind::Player(state.history.len() % 2)
            }
        }

        fn legal_action_count(&self, state: &Self::State) -> usize {
            if state.cards.is_none() {
                return Self::DEALS.len();
            }
            match state.history.as_str() {
                "" | "c" | "b" | "cb" => 2,
                _ => 0,
            }
        }

        fn next_state(&self, state: &Self::State, action_idx: usize) -> Self::State {
            if state.cards.is_none() {
                return KuhnState {
                    cards: Some(Self::DEALS[action_idx]),
                    history: String::new(),
                };
            }
            let mut next = state.clone();
            let action = match state.history.as_str() {
                "" | "c" => ['c', 'b'][action_idx],
                "b" | "cb" => ['f', 'c'][action_idx],
                _ => panic!("invalid non-terminal history {}", state.history),
            };
            next.history.push(action);
            next
        }

        fn terminal_utility(&self, state: &Self::State, player: usize) -> f64 {
            let cards = state.cards.expect("terminal nodes must have cards");
            let u0 = terminal_utility_player0(cards, &state.history).unwrap_or(0.0);
            if player == 0 {
                u0
            } else {
                -u0
            }
        }

        fn infoset_key(&self, state: &Self::State, player: usize) -> String {
            let cards = state.cards.expect("infosets only defined after deal");
            format!("{}:{}", cards[player], state.history)
        }

        fn chance_probabilities(&self, state: &Self::State) -> Vec<f64> {
            if state.cards.is_none() {
                vec![1.0 / Self::DEALS.len() as f64; Self::DEALS.len()]
            } else {
                Vec::new()
            }
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

    #[test]
    fn serial_linear_external_sampling_converges_on_kuhn_ev() {
        let game = KuhnGame;
        let mut trainer = ExternalSamplingTrainer::new(2);
        trainer.train_serial(&game, 350_000, 9);

        let ev = trainer.expected_value_against_average_policy(&game, 0);
        let target = -1.0 / 18.0;
        assert!((ev - target).abs() < 0.02, "ev={ev} target={target}");
    }

    #[test]
    fn exploitability_drops_after_training() {
        let game = KuhnGame;
        let mut trainer = ExternalSamplingTrainer::new(2);
        let start = trainer
            .measure_exploitability_two_player(&game)
            .expect("two-player exploitability should exist")
            .exploitability;
        trainer.train_serial(&game, 80_000, 4);
        let end = trainer
            .measure_exploitability_two_player(&game)
            .expect("two-player exploitability should exist")
            .exploitability;
        assert!(
            end < start,
            "exploitability should decrease: start={start}, end={end}"
        );
    }

    #[test]
    fn infoset_exploitability_near_zero_after_training() {
        let game = KuhnGame;
        let mut trainer = ExternalSamplingTrainer::new(2);
        trainer.train_serial(&game, 500_000, 9);
        let report = trainer
            .measure_infoset_exploitability_two_player(&game)
            .expect("should produce report");
        assert!(
            report.exploitability < 0.05,
            "infoset exploitability too high: {}",
            report.exploitability
        );
    }

    #[test]
    fn sampled_exploitability_matches_full_tree_on_kuhn_when_sampling_all_deals() {
        let game = KuhnGame;
        let mut trainer = ExternalSamplingTrainer::new(2);
        trainer.train_serial(&game, 250_000, 19);

        let full = trainer
            .measure_infoset_exploitability_two_player(&game)
            .expect("full infoset exploitability should exist");
        let sampled = trainer
            .measure_sampled_exploitability_two_player(&game, KuhnGame::DEALS.len(), 1234)
            .expect("sampled exploitability should exist");

        assert!(
            (sampled.exploitability - full.exploitability).abs() < 1e-12,
            "sampled={} full={}",
            sampled.exploitability,
            full.exploitability
        );
        assert!(
            (sampled.expected_values[0] - full.expected_values[0]).abs() < 1e-12,
            "sampled_ev={} full_ev={}",
            sampled.expected_values[0],
            full.expected_values[0]
        );
    }

    #[test]
    fn parallel_training_and_checkpoint_roundtrip() {
        let game = KuhnGame;
        let mut trainer = ExternalSamplingTrainer::new(2);
        trainer.train_parallel(&game, 120_000, 4, 123);
        assert!(trainer.infoset_count() >= 12);

        let path = env::temp_dir().join("wipoker_cfr_checkpoint.bin");
        trainer
            .save_checkpoint(&path)
            .expect("checkpoint save should succeed");
        let loaded = ExternalSamplingTrainer::load_checkpoint(&path)
            .expect("checkpoint load should succeed");
        fs::remove_file(path).ok();

        assert_eq!(loaded.num_players, trainer.num_players);
        assert!(loaded.iterations > 0);
        assert_eq!(loaded.infoset_count(), trainer.infoset_count());
        let avg = loaded.average_policy_table();
        assert_eq!(avg.len(), loaded.infoset_count());
        for probs in avg.values() {
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9 || probs.is_empty());
        }
    }
}
