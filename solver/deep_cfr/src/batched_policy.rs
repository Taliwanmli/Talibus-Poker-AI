use crate::encoding::INPUT_DIM;
use crate::onnx_policy::{OnnxPolicyError, PolicyOutput};
use crate::sample::MAX_ACTIONS;
use crate::traverse::PolicyProvider;
use crossbeam_channel::{bounded, Receiver, Sender};
use ndarray::Array2;
use ort::ep;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct BatchRuntimeConfig {
    pub max_batch_size: usize,
    pub max_wait_us: u64,
    pub queue_capacity: usize,
    pub use_cuda: bool,
    pub cuda_device_id: i32,
    pub cuda_tf32: bool,
}

impl Default for BatchRuntimeConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 256,
            max_wait_us: 500,
            queue_capacity: 8192,
            use_cuda: false,
            cuda_device_id: 0,
            cuda_tf32: true,
        }
    }
}

#[derive(Debug)]
struct InferenceRequest {
    features: [f32; INPUT_DIM],
    action_mask: [f32; MAX_ACTIONS],
    response_tx: Sender<std::result::Result<[f32; MAX_ACTIONS], String>>,
}

#[derive(Debug)]
pub struct BatchInferenceServer {
    sender: Option<Sender<InferenceRequest>>,
    join_handle: Option<thread::JoinHandle<()>>,
}

impl BatchInferenceServer {
    pub fn spawn(
        model_path: PathBuf,
        output_mode: PolicyOutput,
        runtime: BatchRuntimeConfig,
    ) -> std::result::Result<Self, OnnxPolicyError> {
        let capacity = runtime.queue_capacity.max(runtime.max_batch_size.max(1));
        let (sender, receiver) = bounded::<InferenceRequest>(capacity);
        let thread_name = format!("deep-cfr-batch-{}", output_mode.as_str());
        let join_handle = thread::Builder::new()
            .name(thread_name)
            .spawn(move || {
                let _ = run_server_loop(&model_path, receiver, &runtime);
            })
            .map_err(|err| {
                OnnxPolicyError::Message(format!("failed to spawn batch server: {err}"))
            })?;
        Ok(Self {
            sender: Some(sender),
            join_handle: Some(join_handle),
        })
    }

    pub fn client(&self, output_mode: PolicyOutput) -> BatchedOnnxPolicy {
        BatchedOnnxPolicy {
            output_mode,
            sender: self
                .sender
                .as_ref()
                .expect("batch inference server sender should exist")
                .clone(),
        }
    }
}

impl Drop for BatchInferenceServer {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}

#[derive(Debug, Clone)]
pub struct BatchedOnnxPolicy {
    output_mode: PolicyOutput,
    sender: Sender<InferenceRequest>,
}

impl PolicyProvider for BatchedOnnxPolicy {
    fn get_strategy(
        &mut self,
        features: &[f32],
        action_slots: &[usize],
        action_mask: &[f32; MAX_ACTIONS],
    ) -> std::result::Result<Vec<f64>, OnnxPolicyError> {
        if action_slots.is_empty() || action_slots.len() > MAX_ACTIONS {
            return Err(OnnxPolicyError::InvalidActionCount(action_slots.len()));
        }
        let mut seen = [false; MAX_ACTIONS];
        for slot in action_slots {
            if *slot >= MAX_ACTIONS {
                return Err(OnnxPolicyError::InvalidActionSlot(*slot));
            }
            if seen[*slot] {
                return Err(OnnxPolicyError::DuplicateActionSlot(*slot));
            }
            seen[*slot] = true;
        }

        let feature_array: [f32; INPUT_DIM] = match features.try_into() {
            Ok(array) => array,
            Err(_) => return Ok(uniform_strategy(action_slots.len())),
        };
        let (response_tx, response_rx) = bounded(1);
        self.sender
            .send(InferenceRequest {
                features: feature_array,
                action_mask: *action_mask,
                response_tx,
            })
            .map_err(|err| OnnxPolicyError::Message(format!("batch request send failed: {err}")))?;
        let output = response_rx
            .recv()
            .map_err(|err| OnnxPolicyError::Message(format!("batch response recv failed: {err}")))?
            .map_err(OnnxPolicyError::Message)?;
        let action_values = action_slots
            .iter()
            .map(|slot| output[*slot])
            .collect::<Vec<_>>();
        let strategy = match self.output_mode {
            PolicyOutput::Advantage => regret_match(&action_values),
            PolicyOutput::Strategy => normalize_probabilities(&action_values),
        };
        Ok(strategy)
    }
}

fn run_server_loop(
    model_path: &Path,
    receiver: Receiver<InferenceRequest>,
    runtime: &BatchRuntimeConfig,
) -> std::result::Result<(), OnnxPolicyError> {
    let mut session = build_session(model_path, runtime)?;
    let max_batch = runtime.max_batch_size.max(1);
    let timeout = Duration::from_micros(runtime.max_wait_us.max(1));

    loop {
        let first = match receiver.recv() {
            Ok(request) => request,
            Err(_) => break,
        };
        let mut pending = Vec::with_capacity(max_batch);
        pending.push(first);
        let deadline = Instant::now() + timeout;

        while pending.len() < max_batch {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            let wait = deadline.saturating_duration_since(now);
            match receiver.recv_timeout(wait) {
                Ok(request) => pending.push(request),
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => break,
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
            }
        }

        match forward_batch(&mut session, &pending) {
            Ok(outputs) => {
                for (request, output) in pending.into_iter().zip(outputs.into_iter()) {
                    let _ = request.response_tx.send(Ok(output));
                }
            }
            Err(err) => {
                let msg = err.to_string();
                for request in pending {
                    let _ = request.response_tx.send(Err(msg.clone()));
                }
            }
        }
    }
    Ok(())
}

fn build_session(
    model_path: &Path,
    runtime: &BatchRuntimeConfig,
) -> std::result::Result<Session, OnnxPolicyError> {
    let mut builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?;
    if runtime.use_cuda {
        builder = builder.with_execution_providers([
            ep::CUDA::default()
                .with_device_id(runtime.cuda_device_id)
                .with_tf32(runtime.cuda_tf32)
                .build()
                .error_on_failure(),
            ep::CPU::default().build(),
        ])?;
    }
    let session = builder.commit_from_file(model_path)?;
    Ok(session)
}

fn forward_batch(
    session: &mut Session,
    requests: &[InferenceRequest],
) -> std::result::Result<Vec<[f32; MAX_ACTIONS]>, OnnxPolicyError> {
    let batch_size = requests.len().max(1);
    let mut flat_inputs = Vec::with_capacity(batch_size * INPUT_DIM);
    let mut flat_masks = Vec::with_capacity(batch_size * MAX_ACTIONS);
    for request in requests {
        flat_inputs.extend_from_slice(&request.features);
        flat_masks.extend_from_slice(&request.action_mask);
    }

    let input = Array2::from_shape_vec((batch_size, INPUT_DIM), flat_inputs)?;
    let input_tensor = TensorRef::from_array_view(input.view())?;
    let outputs = if session.inputs().len() >= 2 {
        let mask = Array2::from_shape_vec((batch_size, MAX_ACTIONS), flat_masks)?;
        let mask_tensor = TensorRef::from_array_view(mask.view())?;
        session.run(ort::inputs![input_tensor, mask_tensor])?
    } else {
        session.run(ort::inputs![input_tensor])?
    };
    let output = outputs[0].try_extract_array::<f32>()?;
    let output_shape = output.shape();
    let row_stride = output_shape.last().copied().unwrap_or(MAX_ACTIONS);
    if row_stride < MAX_ACTIONS {
        return Err(OnnxPolicyError::Message(format!(
            "invalid batched output row stride: {row_stride}"
        )));
    }
    let values = output.iter().copied().collect::<Vec<_>>();
    let mut all = Vec::with_capacity(batch_size);
    for row_idx in 0..batch_size {
        let base = row_idx.saturating_mul(row_stride);
        if base + MAX_ACTIONS > values.len() {
            return Err(OnnxPolicyError::Message(format!(
                "batched output length mismatch: expected at least {}, got {}",
                base + MAX_ACTIONS,
                values.len()
            )));
        }
        let mut row = [0.0f32; MAX_ACTIONS];
        row.copy_from_slice(&values[base..base + MAX_ACTIONS]);
        all.push(row);
    }
    Ok(all)
}

fn regret_match(advantages: &[f32]) -> Vec<f64> {
    let mut positive = vec![0.0f64; advantages.len()];
    let mut sum = 0.0f64;
    for (idx, advantage) in advantages.iter().enumerate() {
        let value = (*advantage).max(0.0) as f64;
        positive[idx] = value;
        sum += value;
    }
    if sum > 1e-12 {
        positive.iter().map(|v| v / sum).collect()
    } else {
        uniform_strategy(advantages.len())
    }
}

fn normalize_probabilities(probabilities: &[f32]) -> Vec<f64> {
    let mut normalized = vec![0.0f64; probabilities.len()];
    let mut sum = 0.0f64;
    for (idx, value) in probabilities.iter().enumerate() {
        let clamped = if value.is_finite() {
            (*value).max(0.0) as f64
        } else {
            0.0
        };
        normalized[idx] = clamped;
        sum += clamped;
    }
    if sum > 1e-12 {
        normalized.iter().map(|v| v / sum).collect()
    } else {
        uniform_strategy(probabilities.len())
    }
}

fn uniform_strategy(valid_actions: usize) -> Vec<f64> {
    vec![1.0 / valid_actions as f64; valid_actions]
}
