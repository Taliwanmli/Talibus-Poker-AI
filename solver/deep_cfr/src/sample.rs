use crate::encoding::INPUT_DIM;
use cfr::nlhe_game::POLICY_MAX_ACTIONS;
use rand::Rng;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

pub const ADVANTAGE_SAMPLE_MAGIC: [u8; 4] = *b"DCFR";
pub const STRATEGY_SAMPLE_MAGIC: [u8; 4] = *b"DCSG";
pub const SAMPLE_MAGIC: [u8; 4] = ADVANTAGE_SAMPLE_MAGIC;
pub const SAMPLE_VERSION: u32 = 2;
pub const MAX_ACTIONS: usize = POLICY_MAX_ACTIONS;
pub const SAMPLE_HEADER_BYTES: usize = 16;
pub const SAMPLE_BYTES: usize = INPUT_DIM * 4 + MAX_ACTIONS * 4 + MAX_ACTIONS + 4;

#[derive(Clone, Debug, PartialEq)]
pub struct SampleHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub input_dim: u32,
    pub max_actions: u32,
}

impl Default for SampleHeader {
    fn default() -> Self {
        Self::for_magic(ADVANTAGE_SAMPLE_MAGIC)
    }
}

impl SampleHeader {
    pub fn for_magic(magic: [u8; 4]) -> Self {
        Self {
            magic,
            version: SAMPLE_VERSION,
            input_dim: INPUT_DIM as u32,
            max_actions: MAX_ACTIONS as u32,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdvantageSample {
    pub input: [f32; INPUT_DIM],
    pub advantages: [f32; MAX_ACTIONS],
    pub action_mask: [u8; MAX_ACTIONS],
    pub iteration: u32,
}

impl AdvantageSample {
    pub fn new(
        input: [f32; INPUT_DIM],
        advantages: [f32; MAX_ACTIONS],
        action_mask: [u8; MAX_ACTIONS],
        iteration: u32,
    ) -> Self {
        let mut normalized_mask = action_mask;
        normalize_action_mask(&mut normalized_mask);
        Self {
            input,
            advantages,
            action_mask: normalized_mask,
            iteration,
        }
    }
}

pub fn write_samples(path: impl AsRef<Path>, samples: &[AdvantageSample]) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    write_header(
        &mut writer,
        &SampleHeader::for_magic(ADVANTAGE_SAMPLE_MAGIC),
    )?;
    for sample in samples {
        write_advantage_sample(&mut writer, sample)?;
    }
    writer.flush()
}

pub fn read_samples(path: impl AsRef<Path>) -> io::Result<Vec<AdvantageSample>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let header = read_header(&mut reader)?;
    validate_header(&header, ADVANTAGE_SAMPLE_MAGIC)?;
    let version = header.version;

    let mut samples = Vec::new();
    loop {
        if reader.fill_buf()?.is_empty() {
            break;
        }
        samples.push(read_advantage_sample(&mut reader, version)?);
    }
    Ok(samples)
}

pub fn append_samples(path: impl AsRef<Path>, samples: &[AdvantageSample]) -> io::Result<()> {
    let path = path.as_ref();
    if !path.exists() {
        return write_samples(path, samples);
    }

    let mut existing = read_samples(path)?;
    existing.extend_from_slice(samples);
    write_samples(path, &existing)
}

#[derive(Clone, Debug, PartialEq)]
pub struct StrategySample {
    pub input: [f32; INPUT_DIM],
    pub strategy: [f32; MAX_ACTIONS],
    pub action_mask: [u8; MAX_ACTIONS],
    pub iteration: u32,
}

impl StrategySample {
    pub fn new(
        input: [f32; INPUT_DIM],
        strategy: [f32; MAX_ACTIONS],
        action_mask: [u8; MAX_ACTIONS],
        iteration: u32,
    ) -> Self {
        let mut normalized_mask = action_mask;
        normalize_action_mask(&mut normalized_mask);
        Self {
            input,
            strategy,
            action_mask: normalized_mask,
            iteration,
        }
    }
}

pub fn write_strategy_samples(
    path: impl AsRef<Path>,
    samples: &[StrategySample],
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    write_header(&mut writer, &SampleHeader::for_magic(STRATEGY_SAMPLE_MAGIC))?;
    for sample in samples {
        write_strategy_sample(&mut writer, sample)?;
    }
    writer.flush()
}

pub fn read_strategy_samples(path: impl AsRef<Path>) -> io::Result<Vec<StrategySample>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let header = read_header(&mut reader)?;
    validate_header(&header, STRATEGY_SAMPLE_MAGIC)?;
    let version = header.version;

    let mut samples = Vec::new();
    loop {
        if reader.fill_buf()?.is_empty() {
            break;
        }
        samples.push(read_strategy_sample(&mut reader, version)?);
    }
    Ok(samples)
}

pub fn append_strategy_samples(
    path: impl AsRef<Path>,
    samples: &[StrategySample],
) -> io::Result<()> {
    let path = path.as_ref();
    if !path.exists() {
        return write_strategy_samples(path, samples);
    }

    let mut existing = read_strategy_samples(path)?;
    existing.extend_from_slice(samples);
    write_strategy_samples(path, &existing)
}

pub fn write_header(mut writer: impl Write, header: &SampleHeader) -> io::Result<()> {
    writer.write_all(&header.magic)?;
    writer.write_all(&header.version.to_le_bytes())?;
    writer.write_all(&header.input_dim.to_le_bytes())?;
    writer.write_all(&header.max_actions.to_le_bytes())?;
    Ok(())
}

pub fn read_header(mut reader: impl Read) -> io::Result<SampleHeader> {
    let mut magic = [0u8; 4];
    let mut buf = [0u8; 4];
    reader.read_exact(&mut magic)?;
    reader.read_exact(&mut buf)?;
    let version = u32::from_le_bytes(buf);
    reader.read_exact(&mut buf)?;
    let input_dim = u32::from_le_bytes(buf);
    reader.read_exact(&mut buf)?;
    let max_actions = u32::from_le_bytes(buf);
    Ok(SampleHeader {
        magic,
        version,
        input_dim,
        max_actions,
    })
}

fn validate_header(header: &SampleHeader, expected_magic: [u8; 4]) -> io::Result<()> {
    if header.magic != expected_magic {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid sample magic: expected {:?}, got {:?}",
                expected_magic, header.magic
            ),
        ));
    }
    if header.version == 0 || header.version > SAMPLE_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported sample version: expected <= {}, got {}",
                SAMPLE_VERSION, header.version
            ),
        ));
    }
    if header.input_dim != INPUT_DIM as u32 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "input dimension mismatch: expected {}, got {}",
                INPUT_DIM, header.input_dim
            ),
        ));
    }
    if header.max_actions != MAX_ACTIONS as u32 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "max action mismatch: expected {}, got {}",
                MAX_ACTIONS, header.max_actions
            ),
        ));
    }
    Ok(())
}

fn write_vector(mut writer: impl Write, values: &[f32]) -> io::Result<()> {
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn read_vector<const N: usize>(mut reader: impl Read) -> io::Result<[f32; N]> {
    let mut out = [0.0f32; N];
    let mut fbuf = [0u8; 4];
    for value in &mut out {
        reader.read_exact(&mut fbuf)?;
        *value = f32::from_le_bytes(fbuf);
    }
    Ok(out)
}

fn normalize_action_mask(mask: &mut [u8; MAX_ACTIONS]) {
    for value in mask.iter_mut() {
        *value = u8::from(*value > 0);
    }
}

fn prefix_action_mask(valid_actions: u8) -> [u8; MAX_ACTIONS] {
    let mut mask = [0u8; MAX_ACTIONS];
    let limit = usize::from(valid_actions).min(MAX_ACTIONS);
    for value in mask.iter_mut().take(limit) {
        *value = 1;
    }
    mask
}

fn write_advantage_sample(mut writer: impl Write, sample: &AdvantageSample) -> io::Result<()> {
    write_vector(&mut writer, &sample.input)?;
    write_vector(&mut writer, &sample.advantages)?;
    writer.write_all(&sample.action_mask)?;
    writer.write_all(&sample.iteration.to_le_bytes())?;
    Ok(())
}

fn read_advantage_sample(mut reader: impl Read, version: u32) -> io::Result<AdvantageSample> {
    let input = read_vector::<INPUT_DIM>(&mut reader)?;
    let advantages = read_vector::<MAX_ACTIONS>(&mut reader)?;
    let action_mask = if version == 1 {
        let mut vbuf = [0u8; 1];
        reader.read_exact(&mut vbuf)?;
        prefix_action_mask(vbuf[0])
    } else {
        let mut mask = [0u8; MAX_ACTIONS];
        reader.read_exact(&mut mask)?;
        normalize_action_mask(&mut mask);
        mask
    };
    let mut ibuf = [0u8; 4];
    reader.read_exact(&mut ibuf)?;
    let iteration = u32::from_le_bytes(ibuf);

    Ok(AdvantageSample {
        input,
        advantages,
        action_mask,
        iteration,
    })
}

fn write_strategy_sample(mut writer: impl Write, sample: &StrategySample) -> io::Result<()> {
    write_vector(&mut writer, &sample.input)?;
    write_vector(&mut writer, &sample.strategy)?;
    writer.write_all(&sample.action_mask)?;
    writer.write_all(&sample.iteration.to_le_bytes())?;
    Ok(())
}

fn read_strategy_sample(mut reader: impl Read, version: u32) -> io::Result<StrategySample> {
    let input = read_vector::<INPUT_DIM>(&mut reader)?;
    let strategy = read_vector::<MAX_ACTIONS>(&mut reader)?;
    let action_mask = if version == 1 {
        let mut vbuf = [0u8; 1];
        reader.read_exact(&mut vbuf)?;
        prefix_action_mask(vbuf[0])
    } else {
        let mut mask = [0u8; MAX_ACTIONS];
        reader.read_exact(&mut mask)?;
        normalize_action_mask(&mut mask);
        mask
    };
    let mut ibuf = [0u8; 4];
    reader.read_exact(&mut ibuf)?;
    let iteration = u32::from_le_bytes(ibuf);

    Ok(StrategySample {
        input,
        strategy,
        action_mask,
        iteration,
    })
}

#[derive(Clone, Debug)]
pub struct ReservoirBuffer<T> {
    max_size: usize,
    seen: u64,
    items: Vec<T>,
}

impl<T> ReservoirBuffer<T> {
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            seen: 0,
            items: Vec::with_capacity(max_size),
        }
    }

    pub fn max_size(&self) -> usize {
        self.max_size
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn as_slice(&self) -> &[T] {
        &self.items
    }

    pub fn into_vec(self) -> Vec<T> {
        self.items
    }

    pub fn add(&mut self, item: T, rng: &mut impl Rng) {
        self.seen = self.seen.saturating_add(1);
        if self.max_size == 0 {
            return;
        }
        if self.items.len() < self.max_size {
            self.items.push(item);
            return;
        }

        let idx = rng.gen_range(0..self.seen);
        if idx < self.max_size as u64 {
            self.items[idx as usize] = item;
        }
    }

    pub fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I, rng: &mut impl Rng) {
        for item in iter {
            self.add(item, rng);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn write_then_read_round_trips_samples() {
        let mut input = [0.0f32; INPUT_DIM];
        input[0] = 1.0;
        input[10] = 0.5;
        let mut advantages = [0.0f32; MAX_ACTIONS];
        advantages[2] = 1.25;
        advantages[4] = -0.5;
        let mut mask = [0u8; MAX_ACTIONS];
        for value in mask.iter_mut().take(5) {
            *value = 1;
        }
        let sample = AdvantageSample::new(input, advantages, mask, 17);

        let path = std::env::temp_dir().join("deep_cfr_samples_roundtrip.bin");
        write_samples(&path, &[sample.clone()]).expect("should write samples");
        let loaded = read_samples(&path).expect("should read samples");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0], sample);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn write_then_read_round_trips_strategy_samples() {
        let mut input = [0.0f32; INPUT_DIM];
        input[1] = 1.0;
        input[7] = 0.25;
        let mut strategy = [0.0f32; MAX_ACTIONS];
        strategy[0] = 0.2;
        strategy[1] = 0.3;
        strategy[2] = 0.5;
        let mut mask = [0u8; MAX_ACTIONS];
        for value in mask.iter_mut().take(3) {
            *value = 1;
        }
        let sample = StrategySample::new(input, strategy, mask, 9);

        let path = std::env::temp_dir().join("deep_cfr_strategy_samples_roundtrip.bin");
        write_strategy_samples(&path, &[sample.clone()]).expect("should write strategy samples");
        let loaded = read_strategy_samples(&path).expect("should read strategy samples");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0], sample);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn write_then_read_round_trips_non_prefix_action_mask() {
        let mut input = [0.0f32; INPUT_DIM];
        input[2] = 1.0;
        let mut strategy = [0.0f32; MAX_ACTIONS];
        strategy[1] = 0.25;
        strategy[7] = 0.50;
        strategy[9] = 0.25;
        let mut mask = [0u8; MAX_ACTIONS];
        mask[1] = 1;
        mask[7] = 1;
        mask[9] = 1;
        let sample = StrategySample::new(input, strategy, mask, 11);

        let path = std::env::temp_dir().join("deep_cfr_strategy_non_prefix_mask.bin");
        write_strategy_samples(&path, &[sample.clone()]).expect("should write strategy samples");
        let loaded = read_strategy_samples(&path).expect("should read strategy samples");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0], sample);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn stale_max_actions_header_is_rejected() {
        let path = std::env::temp_dir().join("deep_cfr_strategy_stale_max_actions.bin");
        let mut writer = std::io::BufWriter::new(std::fs::File::create(&path).expect("create file"));
        let mut header = SampleHeader::for_magic(STRATEGY_SAMPLE_MAGIC);
        header.max_actions = header.max_actions.saturating_sub(1);
        write_header(&mut writer, &header).expect("write header");
        writer.flush().expect("flush header");

        let err = read_strategy_samples(&path).expect_err("stale max_actions header should fail");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("max action mismatch"),
            "unexpected error: {err}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reservoir_buffer_never_exceeds_max_size() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let mut reservoir = ReservoirBuffer::new(8);
        for idx in 0..1000 {
            reservoir.add(idx, &mut rng);
        }
        assert_eq!(reservoir.len(), 8);
    }
}
