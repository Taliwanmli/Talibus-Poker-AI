use crate::batched_policy::{BatchInferenceServer, BatchRuntimeConfig, BatchedOnnxPolicy};
use crate::encoding::encode_nlhe_state;
use crate::onnx_policy::{OnnxPolicy, OnnxPolicyError, PolicyOutput};
use crate::sample::MAX_ACTIONS;
use crate::traverse::PolicyProvider;
use cfr::external_sampling::{GameModel, NodeKind};
use cfr::nlhe_game::{enumerate_action_space, IndexedAction, NlheGameModel, NlheState};
use dashmap::DashMap;
use game::BettingRound;
use rand::prelude::{Rng, SeedableRng, StdRng};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub type RealtimeResult<T> = Result<T, String>;

const CONTINUATION_ACTIONS: usize = 4;
const CONTINUATION_FOLD_BIAS: f64 = 5.0;
const CONTINUATION_CALL_BIAS: f64 = 5.0;
const CONTINUATION_RAISE_BIAS: f64 = 5.0;

#[derive(Clone, Debug)]
pub struct RealtimeConfig {
    pub time_budget_ms: u64,
    pub worker_threads: usize,
    pub max_iterations: usize,
    pub leaf_rollouts_per_action: usize,
    pub enable_gpu_batch: bool,
    pub rng_seed: u64,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            time_budget_ms: 10_000,
            worker_threads: 24,
            max_iterations: 0,
            leaf_rollouts_per_action: 1,
            enable_gpu_batch: true,
            rng_seed: 42,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SearchActionProbability {
    pub action_index: usize,
    pub action_token: String,
    pub policy_slot: usize,
    pub probability: f64,
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub hero_seat: usize,
    pub elapsed: Duration,
    pub iterations: usize,
    pub infoset_count: usize,
    pub chosen_action_index: usize,
    pub chosen_action_token: String,
    pub action_probabilities: Vec<SearchActionProbability>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContinuationStrategy {
    Blueprint,
    FoldBiased,
    CallBiased,
    RaiseBiased,
}

impl ContinuationStrategy {
    pub fn from_index(value: usize) -> Self {
        match value {
            0 => Self::Blueprint,
            1 => Self::FoldBiased,
            2 => Self::CallBiased,
            3 => Self::RaiseBiased,
            _ => Self::Blueprint,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SearchPathState {
    pub raises_since_root: u8,
}

impl SearchPathState {
    fn with_action(self, action: &IndexedAction) -> Self {
        let mut out = self;
        // sort_group=4 are raises in the abstract action ordering.
        if action.sort_group == 4 {
            out.raises_since_root = out.raises_since_root.saturating_add(1);
        }
        out
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DepthLimiter {
    root_round: BettingRound,
    root_active_players: usize,
}

impl DepthLimiter {
    pub fn from_root_state(root_state: &NlheState) -> RealtimeResult<Self> {
        let game_state = root_state
            .game
            .as_ref()
            .ok_or_else(|| "depth limiter requires a resolved NLHE game state".to_string())?;
        Ok(Self {
            root_round: game_state.round,
            root_active_players: active_player_count(root_state),
        })
    }

    pub fn should_stop(&self, state: &NlheState, path: SearchPathState) -> bool {
        let Some(game_state) = state.game.as_ref() else {
            return false;
        };
        if game_state.is_complete() {
            return false;
        }
        let current_round = game_state.round;
        match self.root_round {
            BettingRound::Preflop => current_round != BettingRound::Preflop,
            BettingRound::Flop if self.root_active_players > 2 => {
                current_round != BettingRound::Flop || path.raises_since_root >= 2
            }
            _ => false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SubgameInfoset {
    pub regrets: Vec<f64>,
    pub strategy_sum: Vec<f64>,
    pub visits: u64,
}

impl SubgameInfoset {
    fn new(action_count: usize) -> Self {
        Self {
            regrets: vec![0.0; action_count],
            strategy_sum: vec![0.0; action_count],
            visits: 0,
        }
    }
}

#[derive(Clone)]
struct SharedInfosets {
    map: Arc<DashMap<String, SubgameInfoset>>,
}

impl SharedInfosets {
    fn new() -> Self {
        Self {
            map: Arc::new(DashMap::new()),
        }
    }

    fn get_strategy(&self, key: &str, action_count: usize) -> Vec<f64> {
        let mut entry = self
            .map
            .entry(key.to_string())
            .or_insert_with(|| SubgameInfoset::new(action_count));
        if entry.regrets.len() != action_count {
            *entry = SubgameInfoset::new(action_count);
        }
        let strategy = regret_match(&entry.regrets);
        for (idx, value) in strategy.iter().enumerate() {
            entry.strategy_sum[idx] += *value;
        }
        entry.visits = entry.visits.saturating_add(1);
        strategy
    }

    fn add_regrets(&self, key: &str, action_count: usize, deltas: &[f64]) {
        if deltas.len() != action_count {
            return;
        }
        let mut entry = self
            .map
            .entry(key.to_string())
            .or_insert_with(|| SubgameInfoset::new(action_count));
        if entry.regrets.len() != action_count {
            *entry = SubgameInfoset::new(action_count);
        }
        for (idx, delta) in deltas.iter().enumerate() {
            entry.regrets[idx] += *delta;
        }
    }

    fn average_strategy(&self, key: &str, action_count: usize) -> Option<Vec<f64>> {
        let entry = self.map.get(key)?;
        if entry.strategy_sum.len() != action_count {
            return None;
        }
        Some(normalize_probs(&entry.strategy_sum))
    }

    fn current_strategy(&self, key: &str, action_count: usize) -> Option<Vec<f64>> {
        let entry = self.map.get(key)?;
        if entry.regrets.len() != action_count {
            return None;
        }
        Some(regret_match(&entry.regrets))
    }

    fn apply_linear_discount(&self, factor: f64) {
        if !factor.is_finite() || !(0.0..=1.0).contains(&factor) {
            return;
        }
        for mut entry in self.map.iter_mut() {
            let value = entry.value_mut();
            for regret in &mut value.regrets {
                *regret *= factor;
            }
            for strategy_sum in &mut value.strategy_sum {
                *strategy_sum *= factor;
            }
        }
    }

    fn len(&self) -> usize {
        self.map.len()
    }
}

enum WorkerPolicy {
    Batched(BatchedOnnxPolicy),
    Onnx(OnnxPolicy),
}

impl WorkerPolicy {
    fn get_strategy(
        &mut self,
        features: &[f32],
        action_slots: &[usize],
        action_mask: &[f32; MAX_ACTIONS],
    ) -> RealtimeResult<Vec<f64>> {
        match self {
            Self::Batched(policy) => policy
                .get_strategy(features, action_slots, action_mask)
                .map_err(|err| format!("batched policy query failed: {err}")),
            Self::Onnx(policy) => policy
                .get_strategy(features, action_slots, action_mask)
                .map_err(|err| format!("onnx policy query failed: {err}")),
        }
    }
}

struct CachedExecutionResources {
    thread_count: usize,
    use_batch_inference: bool,
    thread_pool: Arc<ThreadPool>,
    worker_policies: Vec<WorkerPolicy>,
}

pub struct RealtimeSearcher {
    model_path: PathBuf,
    output_mode: PolicyOutput,
    batch_runtime: BatchRuntimeConfig,
    batch_server: Mutex<Option<BatchInferenceServer>>,
    execution_resources: Mutex<Option<CachedExecutionResources>>,
}

impl RealtimeSearcher {
    pub fn from_model(path: impl AsRef<Path>) -> Self {
        Self {
            model_path: path.as_ref().to_path_buf(),
            output_mode: PolicyOutput::Strategy,
            batch_runtime: BatchRuntimeConfig {
                max_batch_size: 128,
                max_wait_us: 50,
                queue_capacity: 8_192,
                use_cuda: true,
                cuda_device_id: 0,
                cuda_tf32: true,
            },
            batch_server: Mutex::new(None),
            execution_resources: Mutex::new(None),
        }
    }

    pub fn with_output_mode(mut self, output_mode: PolicyOutput) -> Self {
        self.output_mode = output_mode;
        self
    }

    pub fn with_batch_runtime(mut self, runtime: BatchRuntimeConfig) -> Self {
        self.batch_runtime = runtime;
        self
    }

    fn build_worker_policies(
        &self,
        thread_count: usize,
        use_batch_inference: bool,
    ) -> RealtimeResult<Vec<WorkerPolicy>> {
        let mut worker_policies = Vec::with_capacity(thread_count);
        if use_batch_inference {
            let mut guard = self
                .batch_server
                .lock()
                .expect("batch server mutex should not be poisoned");
            if guard.is_none() {
                let server = BatchInferenceServer::spawn(
                    self.model_path.clone(),
                    self.output_mode,
                    self.batch_runtime.clone(),
                )
                .map_err(|err| format!("failed to start GPU batch inference server: {err}"))?;
                *guard = Some(server);
            }
            let server = guard.as_ref().ok_or_else(|| {
                "batch server unexpectedly missing after initialization".to_string()
            })?;
            for _ in 0..thread_count {
                worker_policies.push(WorkerPolicy::Batched(server.client(self.output_mode)));
            }
            return Ok(worker_policies);
        }

        for _ in 0..thread_count {
            let policy = OnnxPolicy::from_file_with_output_mode(&self.model_path, self.output_mode)
                .map_err(|err| {
                    format!(
                        "failed to load onnx model {}: {err}",
                        self.model_path.display()
                    )
                })?;
            worker_policies.push(WorkerPolicy::Onnx(policy));
        }
        Ok(worker_policies)
    }

    pub fn search(
        &self,
        model: &NlheGameModel,
        root_state: &NlheState,
        hero_seat: usize,
        cfg: &RealtimeConfig,
    ) -> RealtimeResult<SearchResult> {
        let thread_count = cfg.worker_threads.max(1);
        let (thread_pool, mut worker_policies) = {
            let mut resources_guard = self
                .execution_resources
                .lock()
                .expect("execution resources mutex should not be poisoned");
            let needs_rebuild = resources_guard.as_ref().map_or(true, |resources| {
                resources.thread_count != thread_count
                    || resources.use_batch_inference != cfg.enable_gpu_batch
            });
            if needs_rebuild {
                let thread_pool = Arc::new(
                    ThreadPoolBuilder::new()
                        .num_threads(thread_count)
                        .build()
                        .map_err(|err| format!("failed to build thread pool: {err}"))?,
                );
                let worker_policies =
                    self.build_worker_policies(thread_count, cfg.enable_gpu_batch)?;
                *resources_guard = Some(CachedExecutionResources {
                    thread_count,
                    use_batch_inference: cfg.enable_gpu_batch,
                    thread_pool,
                    worker_policies,
                });
            }
            let resources = resources_guard.as_mut().ok_or_else(|| {
                "execution resources unexpectedly missing after initialization".to_string()
            })?;
            (
                Arc::clone(&resources.thread_pool),
                std::mem::take(&mut resources.worker_policies),
            )
        };

        let run_result = run_search(
            model,
            root_state,
            hero_seat,
            cfg,
            thread_pool.as_ref(),
            &mut worker_policies,
        );

        let mut resources_guard = self
            .execution_resources
            .lock()
            .expect("execution resources mutex should not be poisoned");
        if let Some(resources) = resources_guard.as_mut() {
            if resources.thread_count == thread_count
                && resources.use_batch_inference == cfg.enable_gpu_batch
                && resources.worker_policies.is_empty()
            {
                resources.worker_policies = worker_policies;
            }
        }

        run_result
    }
}

fn run_search(
    model: &NlheGameModel,
    root_state: &NlheState,
    hero_seat: usize,
    cfg: &RealtimeConfig,
    thread_pool: &ThreadPool,
    worker_policies: &mut [WorkerPolicy],
) -> RealtimeResult<SearchResult> {
    let actor = match model.node_kind(root_state) {
        NodeKind::Player(idx) => idx,
        NodeKind::Terminal => return Err("search root cannot be terminal".to_string()),
        NodeKind::Chance => return Err("search root cannot be a chance node".to_string()),
    };
    if actor != hero_seat {
        return Err(format!(
            "hero seat mismatch: root actor={actor}, requested hero_seat={hero_seat}"
        ));
    }

    let game_state = root_state
        .game
        .as_ref()
        .ok_or_else(|| "search root must contain a resolved NLHE game state".to_string())?;
    let root_action_space = enumerate_action_space(game_state, hero_seat);
    if root_action_space.is_empty() {
        return Err("search root has no legal actions".to_string());
    }
    let action_count = root_action_space.len();
    let depth_limiter = DepthLimiter::from_root_state(root_state)?;
    let infosets = SharedInfosets::new();

    if worker_policies.is_empty() {
        return Err("run_search requires at least one worker policy".to_string());
    }
    let max_iterations = if cfg.max_iterations == 0 {
        usize::MAX
    } else {
        cfg.max_iterations
    };
    let deadline = Instant::now() + Duration::from_millis(cfg.time_budget_ms.max(1));
    let start = Instant::now();

    let iteration_counter = Arc::new(AtomicUsize::new(0));
    let root_clone = root_state.clone();
    let cycle_len = model.config.num_players.max(1);

    thread_pool.scope(|scope| {
        for (worker_idx, mut worker_policy) in worker_policies.iter_mut().enumerate() {
            let local_infosets = infosets.clone();
            let local_counter = Arc::clone(&iteration_counter);
            let local_root = root_clone.clone();
            scope.spawn(move |_| {
                let mut rng = StdRng::seed_from_u64(
                    cfg.rng_seed ^ ((worker_idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                );
                loop {
                    if Instant::now() >= deadline {
                        break;
                    }
                    let iter_idx = local_counter.fetch_add(1, AtomicOrdering::Relaxed);
                    if iter_idx >= max_iterations {
                        break;
                    }
                    let traverser = iter_idx % model.config.num_players;
                    let _ = traverse_subgame(
                        model,
                        &local_root,
                        traverser,
                        &local_infosets,
                        &mut worker_policy,
                        cfg,
                        depth_limiter,
                        SearchPathState::default(),
                        &mut rng,
                    );
                    let completed_iterations = iter_idx.saturating_add(1);
                    if completed_iterations % cycle_len == 0 {
                        let t = (completed_iterations / cycle_len) as f64;
                        if t > 0.0 {
                            let factor = t / (t + 1.0);
                            local_infosets.apply_linear_discount(factor);
                        }
                    }
                }
            });
        }
    });

    let iterations = iteration_counter
        .load(AtomicOrdering::Relaxed)
        .min(max_iterations);
    let root_infoset_key = model.infoset_key(root_state, hero_seat);
    let strategy = infosets
        .current_strategy(&root_infoset_key, action_count)
        .or_else(|| infosets.average_strategy(&root_infoset_key, action_count))
        .unwrap_or_else(|| vec![1.0 / action_count as f64; action_count]);
    let normalized = normalize_probs(&strategy);

    let mut chosen_action_index = 0usize;
    let mut best_prob = f64::NEG_INFINITY;
    let mut action_probabilities = Vec::with_capacity(action_count);
    for (idx, action) in root_action_space.iter().enumerate() {
        let prob = normalized.get(idx).copied().unwrap_or(0.0);
        if prob > best_prob {
            best_prob = prob;
            chosen_action_index = idx;
        }
        action_probabilities.push(SearchActionProbability {
            action_index: idx,
            action_token: action.action_token.clone(),
            policy_slot: usize::from(action.policy_slot),
            probability: prob,
        });
    }
    let chosen_action_token = root_action_space
        .get(chosen_action_index)
        .map(|action| action.action_token.clone())
        .unwrap_or_else(|| "unknown".to_string());

    Ok(SearchResult {
        hero_seat,
        elapsed: start.elapsed(),
        iterations,
        infoset_count: infosets.len(),
        chosen_action_index,
        chosen_action_token,
        action_probabilities,
    })
}

fn traverse_subgame(
    model: &NlheGameModel,
    state: &NlheState,
    traverser: usize,
    infosets: &SharedInfosets,
    worker_policy: &mut WorkerPolicy,
    cfg: &RealtimeConfig,
    depth_limiter: DepthLimiter,
    path_state: SearchPathState,
    rng: &mut StdRng,
) -> RealtimeResult<f64> {
    if matches!(model.node_kind(state), NodeKind::Terminal) {
        return Ok(model.terminal_utility(state, traverser));
    }
    if depth_limiter.should_stop(state, path_state) {
        return evaluate_leaf(model, state, traverser, infosets, worker_policy, cfg, rng);
    }

    match model.node_kind(state) {
        NodeKind::Terminal => Ok(model.terminal_utility(state, traverser)),
        NodeKind::Chance => {
            let probs = model.chance_probabilities(state);
            let sampled = sample_from_probs(&probs, rng);
            let next = model.next_state(state, sampled);
            traverse_subgame(
                model,
                &next,
                traverser,
                infosets,
                worker_policy,
                cfg,
                depth_limiter,
                path_state,
                rng,
            )
        }
        NodeKind::Player(player_idx) => {
            let game_state = state
                .game
                .as_ref()
                .ok_or_else(|| "expected resolved NLHE game at player node".to_string())?;
            let action_space = enumerate_action_space(game_state, player_idx);
            if action_space.is_empty() {
                return Ok(model.terminal_utility(state, traverser));
            }
            let infoset_key = model.infoset_key(state, player_idx);
            let strategy = infosets.get_strategy(&infoset_key, action_space.len());
            if player_idx == traverser {
                let mut child_values = vec![0.0f64; action_space.len()];
                for (action_idx, action) in action_space.iter().enumerate() {
                    let next = model.next_state(state, action_idx);
                    child_values[action_idx] = traverse_subgame(
                        model,
                        &next,
                        traverser,
                        infosets,
                        worker_policy,
                        cfg,
                        depth_limiter,
                        path_state.with_action(action),
                        rng,
                    )?;
                }
                let node_value = strategy
                    .iter()
                    .zip(child_values.iter())
                    .map(|(p, v)| p * v)
                    .sum::<f64>();
                let regrets = child_values
                    .iter()
                    .map(|value| value - node_value)
                    .collect::<Vec<_>>();
                infosets.add_regrets(&infoset_key, action_space.len(), &regrets);
                Ok(node_value)
            } else {
                let sampled = sample_from_probs(&strategy, rng);
                let sampled = sampled.min(action_space.len().saturating_sub(1));
                let next = model.next_state(state, sampled);
                traverse_subgame(
                    model,
                    &next,
                    traverser,
                    infosets,
                    worker_policy,
                    cfg,
                    depth_limiter,
                    path_state.with_action(&action_space[sampled]),
                    rng,
                )
            }
        }
    }
}

fn evaluate_leaf(
    model: &NlheGameModel,
    state: &NlheState,
    traverser: usize,
    infosets: &SharedInfosets,
    worker_policy: &mut WorkerPolicy,
    cfg: &RealtimeConfig,
    rng: &mut StdRng,
) -> RealtimeResult<f64> {
    let game_state = state
        .game
        .as_ref()
        .ok_or_else(|| "leaf evaluation requires resolved NLHE state".to_string())?;
    let mut continuation_by_player =
        vec![ContinuationStrategy::Blueprint; model.config.num_players];
    for seat in 0..model.config.num_players {
        let player = &game_state.players[seat];
        if player.folded || player.all_in || seat == traverser {
            continue;
        }
        let key = continuation_infoset_key(model, state, seat);
        let strategy = infosets.get_strategy(&key, CONTINUATION_ACTIONS);
        let sampled = sample_from_probs(&strategy, rng);
        continuation_by_player[seat] = ContinuationStrategy::from_index(sampled);
    }

    let key = continuation_infoset_key(model, state, traverser);
    let strategy = infosets.get_strategy(&key, CONTINUATION_ACTIONS);
    let rollout_count = cfg.leaf_rollouts_per_action.max(1);
    let mut action_values = vec![0.0f64; CONTINUATION_ACTIONS];
    for action_idx in 0..CONTINUATION_ACTIONS {
        continuation_by_player[traverser] = ContinuationStrategy::from_index(action_idx);
        let mut total = 0.0f64;
        for _ in 0..rollout_count {
            let mut rollout_rng = StdRng::seed_from_u64(rng.gen::<u64>());
            total += rollout_to_terminal(
                model,
                state.clone(),
                traverser,
                &continuation_by_player,
                worker_policy,
                &mut rollout_rng,
            )?;
        }
        action_values[action_idx] = total / rollout_count as f64;
    }

    let node_value = strategy
        .iter()
        .zip(action_values.iter())
        .map(|(p, v)| p * v)
        .sum::<f64>();
    let regrets = action_values
        .iter()
        .map(|value| value - node_value)
        .collect::<Vec<_>>();
    infosets.add_regrets(&key, CONTINUATION_ACTIONS, &regrets);
    Ok(node_value)
}

fn rollout_to_terminal(
    model: &NlheGameModel,
    mut state: NlheState,
    traverser: usize,
    continuation_by_player: &[ContinuationStrategy],
    worker_policy: &mut WorkerPolicy,
    rng: &mut StdRng,
) -> RealtimeResult<f64> {
    loop {
        match model.node_kind(&state) {
            NodeKind::Terminal => return Ok(model.terminal_utility(&state, traverser)),
            NodeKind::Chance => {
                let probs = model.chance_probabilities(&state);
                let idx = sample_from_probs(&probs, rng);
                state = model.next_state(&state, idx);
            }
            NodeKind::Player(player_idx) => {
                let game_state = state
                    .game
                    .as_ref()
                    .ok_or_else(|| "expected resolved NLHE state during rollout".to_string())?;
                let action_space = enumerate_action_space(game_state, player_idx);
                if action_space.is_empty() {
                    return Ok(model.terminal_utility(&state, traverser));
                }
                let features = encode_nlhe_state(&state, player_idx)
                    .ok_or_else(|| "failed to encode features for rollout".to_string())?;
                let (action_slots, action_mask) = action_slots_and_mask(&action_space)?;
                let mut strategy =
                    worker_policy.get_strategy(&features, &action_slots, &action_mask)?;
                if strategy.len() != action_space.len() {
                    return Err(format!(
                        "rollout strategy/action mismatch: strategy={}, actions={}",
                        strategy.len(),
                        action_space.len()
                    ));
                }
                let style = continuation_by_player
                    .get(player_idx)
                    .copied()
                    .unwrap_or(ContinuationStrategy::Blueprint);
                apply_continuation_bias(&mut strategy, &action_space, style);
                let action_idx = sample_from_probs(&strategy, rng);
                state = model.next_state(&state, action_idx);
            }
        }
    }
}

fn continuation_infoset_key(model: &NlheGameModel, state: &NlheState, player_idx: usize) -> String {
    format!(
        "cont|p={player_idx}|{}",
        model.infoset_key(state, player_idx)
    )
}

fn apply_continuation_bias(
    probabilities: &mut [f64],
    action_space: &[IndexedAction],
    style: ContinuationStrategy,
) {
    if probabilities.is_empty() || probabilities.len() != action_space.len() {
        return;
    }
    match style {
        ContinuationStrategy::Blueprint => {}
        ContinuationStrategy::FoldBiased => {
            for (idx, action) in action_space.iter().enumerate() {
                if action.sort_group == 0 {
                    probabilities[idx] *= CONTINUATION_FOLD_BIAS;
                }
            }
        }
        ContinuationStrategy::CallBiased => {
            for (idx, action) in action_space.iter().enumerate() {
                if action.sort_group == 1 || action.sort_group == 2 {
                    probabilities[idx] *= CONTINUATION_CALL_BIAS;
                }
            }
        }
        ContinuationStrategy::RaiseBiased => {
            for (idx, action) in action_space.iter().enumerate() {
                if action.sort_group >= 3 {
                    probabilities[idx] *= CONTINUATION_RAISE_BIAS;
                }
            }
        }
    }
    let normalized = normalize_probs(probabilities);
    probabilities.copy_from_slice(&normalized);
}

fn action_slots_and_mask(
    action_space: &[IndexedAction],
) -> RealtimeResult<(Vec<usize>, [f32; MAX_ACTIONS])> {
    let mut slots = Vec::with_capacity(action_space.len());
    let mut mask = [0.0f32; MAX_ACTIONS];
    for action in action_space {
        let slot = usize::from(action.policy_slot);
        if slot >= MAX_ACTIONS {
            return Err(format!("invalid policy slot {slot} in action space"));
        }
        if mask[slot] > 0.0 {
            return Err(format!("duplicate policy slot {slot} in action space"));
        }
        mask[slot] = 1.0;
        slots.push(slot);
    }
    Ok((slots, mask))
}

fn active_player_count(state: &NlheState) -> usize {
    let Some(game_state) = state.game.as_ref() else {
        return 0;
    };
    game_state
        .players
        .iter()
        .filter(|player| !player.folded)
        .count()
}

fn sample_from_probs(probabilities: &[f64], rng: &mut StdRng) -> usize {
    if probabilities.is_empty() {
        return 0;
    }
    let draw = rng.gen::<f64>();
    let mut cumulative = 0.0f64;
    for (idx, prob) in probabilities.iter().enumerate() {
        cumulative += *prob;
        if draw <= cumulative {
            return idx;
        }
    }
    probabilities.len().saturating_sub(1)
}

fn regret_match(regrets: &[f64]) -> Vec<f64> {
    let mut positive = vec![0.0f64; regrets.len()];
    let mut sum = 0.0f64;
    for (idx, value) in regrets.iter().enumerate() {
        let clipped = (*value).max(0.0);
        positive[idx] = clipped;
        sum += clipped;
    }
    if sum > 1e-12 {
        positive.iter().map(|value| value / sum).collect()
    } else if regrets.is_empty() {
        Vec::new()
    } else {
        vec![1.0 / regrets.len() as f64; regrets.len()]
    }
}

fn normalize_probs(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0.0f64; values.len()];
    let mut sum = 0.0f64;
    for (idx, value) in values.iter().enumerate() {
        let clipped = if value.is_finite() {
            (*value).max(0.0)
        } else {
            0.0
        };
        out[idx] = clipped;
        sum += clipped;
    }
    if sum <= 1e-12 {
        vec![1.0 / values.len() as f64; values.len()]
    } else {
        out.iter().map(|value| value / sum).collect()
    }
}

pub fn sample_action_from_result(result: &SearchResult, rng: &mut StdRng) -> usize {
    let probs = result
        .action_probabilities
        .iter()
        .map(|entry| entry.probability)
        .collect::<Vec<_>>();
    sample_from_probs(&probs, rng)
}

pub fn onnx_load_test(
    path: impl AsRef<Path>,
    output_mode: PolicyOutput,
) -> Result<(), OnnxPolicyError> {
    let _ = OnnxPolicy::from_file_with_output_mode(path, output_mode)?;
    Ok(())
}
