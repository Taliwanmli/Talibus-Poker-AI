use crate::encoding::INPUT_DIM;
use crate::sample::MAX_ACTIONS;
use crate::traverse::PolicyProvider;
use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum OnnxPolicyError {
    Ort(ort::Error),
    Shape(ndarray::ShapeError),
    InvalidActionCount(usize),
    InvalidActionSlot(usize),
    DuplicateActionSlot(usize),
    Message(String),
}

impl std::fmt::Display for OnnxPolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ort(err) => write!(f, "{err}"),
            Self::Shape(err) => write!(f, "{err}"),
            Self::InvalidActionCount(count) => write!(f, "invalid action count: {count}"),
            Self::InvalidActionSlot(slot) => write!(f, "invalid action slot: {slot}"),
            Self::DuplicateActionSlot(slot) => write!(f, "duplicate action slot: {slot}"),
            Self::Message(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for OnnxPolicyError {}

impl From<ort::Error> for OnnxPolicyError {
    fn from(value: ort::Error) -> Self {
        Self::Ort(value)
    }
}

impl From<ndarray::ShapeError> for OnnxPolicyError {
    fn from(value: ndarray::ShapeError) -> Self {
        Self::Shape(value)
    }
}

pub type Result<T> = std::result::Result<T, OnnxPolicyError>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyOutput {
    Advantage,
    Strategy,
}

impl PolicyOutput {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "advantage" => Some(Self::Advantage),
            "strategy" => Some(Self::Strategy),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Advantage => "advantage",
            Self::Strategy => "strategy",
        }
    }
}

pub struct OnnxPolicy {
    model_path: PathBuf,
    output_mode: PolicyOutput,
    session: Session,
}

impl OnnxPolicy {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_file_with_output_mode(path, PolicyOutput::Advantage)
    }

    pub fn from_file_with_output_mode(
        path: impl AsRef<Path>,
        output_mode: PolicyOutput,
    ) -> Result<Self> {
        let model_path = path.as_ref().to_path_buf();
        let session = Self::build_session(&model_path)?;
        Ok(Self {
            model_path,
            output_mode,
            session,
        })
    }

    pub fn reload(&mut self) -> Result<()> {
        self.session = Self::build_session(&self.model_path)?;
        Ok(())
    }

    pub fn clone_for_worker(&self) -> Result<Self> {
        Self::from_file_with_output_mode(&self.model_path, self.output_mode)
    }

    pub fn get_strategy(
        &mut self,
        features: &[f32],
        action_slots: &[usize],
        action_mask: &[f32; MAX_ACTIONS],
    ) -> Result<Vec<f64>> {
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
        if features.len() != INPUT_DIM {
            return Ok(uniform_strategy(action_slots.len()));
        }

        let model_output = self.forward(features, action_mask)?;
        let action_values = action_slots
            .iter()
            .map(|slot| model_output[*slot])
            .collect::<Vec<_>>();
        let strategy = match self.output_mode {
            PolicyOutput::Advantage => regret_match(&action_values),
            PolicyOutput::Strategy => normalize_probabilities(&action_values),
        };
        Ok(strategy)
    }

    pub fn forward(
        &mut self,
        features: &[f32],
        action_mask: &[f32; MAX_ACTIONS],
    ) -> Result<[f32; MAX_ACTIONS]> {
        let input = Array2::from_shape_vec((1, INPUT_DIM), features.to_vec())?;
        let input_tensor = TensorRef::from_array_view(input.view())?;
        let outputs = if self.session.inputs().len() >= 2 {
            let mask = Array2::from_shape_vec((1, MAX_ACTIONS), action_mask.to_vec())?;
            let mask_tensor = TensorRef::from_array_view(mask.view())?;
            self.session.run(ort::inputs![input_tensor, mask_tensor])?
        } else {
            // Backward compatibility for older one-input exports.
            self.session.run(ort::inputs![input_tensor])?
        };

        let output = outputs[0].try_extract_array::<f32>()?;
        let mut out = [0.0f32; MAX_ACTIONS];
        for (idx, value) in output.iter().take(MAX_ACTIONS).enumerate() {
            out[idx] = *value;
        }
        Ok(out)
    }

    fn build_session(path: &Path) -> Result<Session> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(path)?;
        Ok(session)
    }
}

fn regret_match(advantages: &[f32]) -> Vec<f64> {
    let mut positive = vec![0.0f64; advantages.len()];
    let mut sum = 0.0f64;

    for idx in 0..advantages.len() {
        let value = advantages[idx].max(0.0) as f64;
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

    for idx in 0..probabilities.len() {
        let value = probabilities[idx];
        let clamped = if value.is_finite() {
            value.max(0.0) as f64
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

impl PolicyProvider for OnnxPolicy {
    fn get_strategy(
        &mut self,
        features: &[f32],
        action_slots: &[usize],
        action_mask: &[f32; MAX_ACTIONS],
    ) -> Result<Vec<f64>> {
        OnnxPolicy::get_strategy(self, features, action_slots, action_mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regret_matching_falls_back_to_uniform_when_non_positive() {
        let mut advantages = vec![0.0f32; MAX_ACTIONS];
        advantages[0] = -1.0;
        advantages[1] = -0.2;
        advantages[2] = 0.0;
        let strategy = regret_match(&advantages[..3]);
        assert_eq!(strategy.len(), 3);
        assert!((strategy[0] - 1.0 / 3.0).abs() < 1e-9);
        assert!((strategy[1] - 1.0 / 3.0).abs() < 1e-9);
        assert!((strategy[2] - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn regret_matching_normalizes_positive_values() {
        let mut advantages = vec![0.0f32; MAX_ACTIONS];
        advantages[0] = 1.0;
        advantages[1] = 3.0;
        advantages[2] = -5.0;
        let strategy = regret_match(&advantages[..3]);
        assert!((strategy[0] - 0.25).abs() < 1e-9);
        assert!((strategy[1] - 0.75).abs() < 1e-9);
        assert!((strategy[2] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn strategy_output_normalizes_and_clamps_invalid_values() {
        let mut probabilities = vec![0.0f32; MAX_ACTIONS];
        probabilities[0] = 0.2;
        probabilities[1] = -0.4;
        probabilities[2] = 0.8;
        probabilities[3] = f32::NAN;
        let strategy = normalize_probabilities(&probabilities[..4]);
        assert_eq!(strategy.len(), 4);
        assert!((strategy[0] - 0.2).abs() < 1e-9);
        assert!((strategy[1] - 0.0).abs() < 1e-9);
        assert!((strategy[2] - 0.8).abs() < 1e-9);
        assert!((strategy[3] - 0.0).abs() < 1e-9);
    }
}
