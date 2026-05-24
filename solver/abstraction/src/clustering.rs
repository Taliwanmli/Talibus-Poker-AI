use rand::prelude::{SeedableRng, SliceRandom, StdRng};
use rs_poker::core::{Card, Value as RsValue};
use std::cmp::Ordering;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

const CLUSTER_FILE_MAGIC: &[u8; 8] = b"WPKCLST1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClusteringError {
    EmptyDataset,
    InvalidK,
    InvalidDimensions,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BucketStreet {
    Preflop,
    Flop,
    Turn,
    River,
}

#[derive(Clone, Debug)]
pub struct KMeansConfig {
    pub k: usize,
    pub max_iters: usize,
    pub seed: u64,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            k: 2,
            max_iters: 100,
            seed: 13,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ClusteringResult {
    pub assignments: Vec<usize>,
    pub centroids: Vec<Vec<f64>>,
    pub iterations_run: usize,
}

pub fn buckets_for_street(street: BucketStreet) -> usize {
    match street {
        BucketStreet::Preflop => 169,
        BucketStreet::Flop => 500,
        BucketStreet::Turn => 500,
        BucketStreet::River => 500,
    }
}

pub fn normalize_histogram(hist: &[f64]) -> Vec<f64> {
    let sum: f64 = hist.iter().sum();
    if sum <= 0.0 {
        vec![0.0; hist.len()]
    } else {
        hist.iter().map(|v| v / sum).collect()
    }
}

pub fn emd_distance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "histograms must have equal length");
    let mut cumulative = 0.0;
    let mut distance = 0.0;
    for idx in 0..a.len() {
        cumulative += a[idx] - b[idx];
        distance += cumulative.abs();
    }
    distance
}

pub fn kmeans_emd(
    data: &[Vec<f64>],
    config: &KMeansConfig,
) -> Result<ClusteringResult, ClusteringError> {
    if data.is_empty() {
        return Err(ClusteringError::EmptyDataset);
    }
    if config.k == 0 || config.k > data.len() {
        return Err(ClusteringError::InvalidK);
    }
    let dims = data[0].len();
    if dims == 0 || data.iter().any(|d| d.len() != dims) {
        return Err(ClusteringError::InvalidDimensions);
    }
    if config.max_iters == 0 {
        return Ok(ClusteringResult {
            assignments: vec![0; data.len()],
            centroids: vec![data[0].clone()],
            iterations_run: 0,
        });
    }

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut centroids = initialize_centroids(data, config.k, &mut rng);
    let mut assignments = vec![0usize; data.len()];
    let mut iterations_run = 0usize;

    for iter in 0..config.max_iters {
        iterations_run = iter + 1;
        let mut changed = false;

        for (idx, point) in data.iter().enumerate() {
            let new_cluster = nearest_centroid(point, &centroids);
            if assignments[idx] != new_cluster {
                assignments[idx] = new_cluster;
                changed = true;
            }
        }

        let mut grouped: Vec<Vec<&[f64]>> = vec![Vec::new(); centroids.len()];
        for (idx, point) in data.iter().enumerate() {
            grouped[assignments[idx]].push(point);
        }

        for cluster_idx in 0..centroids.len() {
            if grouped[cluster_idx].is_empty() {
                let replacement = farthest_point_index(data, &centroids, &assignments);
                centroids[cluster_idx] = data[replacement].clone();
                assignments[replacement] = cluster_idx;
                changed = true;
            } else {
                centroids[cluster_idx] =
                    normalize_histogram(&mean_point(&grouped[cluster_idx], dims));
            }
        }

        if !changed {
            break;
        }
    }

    Ok(ClusteringResult {
        assignments,
        centroids,
        iterations_run,
    })
}

pub fn save_clustering_result(path: &Path, result: &ClusteringResult) -> io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(CLUSTER_FILE_MAGIC)?;

    let n = result.assignments.len() as u32;
    let k = result.centroids.len() as u32;
    let d = if result.centroids.is_empty() {
        0u32
    } else {
        result.centroids[0].len() as u32
    };

    file.write_all(&n.to_le_bytes())?;
    file.write_all(&k.to_le_bytes())?;
    file.write_all(&d.to_le_bytes())?;

    for &assignment in &result.assignments {
        file.write_all(&(assignment as u32).to_le_bytes())?;
    }
    for centroid in &result.centroids {
        for &value in centroid {
            file.write_all(&value.to_le_bytes())?;
        }
    }
    Ok(())
}

pub fn load_clustering_result(path: &Path) -> io::Result<ClusteringResult> {
    let mut file = File::open(path)?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;
    if &magic != CLUSTER_FILE_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid clustering file magic",
        ));
    }

    let n = read_u32(&mut file)? as usize;
    let k = read_u32(&mut file)? as usize;
    let d = read_u32(&mut file)? as usize;

    let mut assignments = vec![0usize; n];
    for slot in &mut assignments {
        *slot = read_u32(&mut file)? as usize;
    }

    let mut centroids = vec![vec![0.0; d]; k];
    for centroid in &mut centroids {
        for value in centroid {
            *value = read_f64(&mut file)?;
        }
    }

    Ok(ClusteringResult {
        assignments,
        centroids,
        iterations_run: 0,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CanonicalPreflopHand {
    Pair(RsValue),
    Suited { high: RsValue, low: RsValue },
    Offsuit { high: RsValue, low: RsValue },
}

pub fn canonicalize_preflop_hand(hand: [Card; 2]) -> CanonicalPreflopHand {
    let (first, second) = order_by_rank_desc(hand[0], hand[1]);
    if first.value == second.value {
        return CanonicalPreflopHand::Pair(first.value);
    }
    let suited = first.suit == second.suit;
    if suited {
        CanonicalPreflopHand::Suited {
            high: first.value,
            low: second.value,
        }
    } else {
        CanonicalPreflopHand::Offsuit {
            high: first.value,
            low: second.value,
        }
    }
}

pub fn canonical_preflop_index(canonical: CanonicalPreflopHand) -> usize {
    let high = match canonical {
        CanonicalPreflopHand::Pair(v)
        | CanonicalPreflopHand::Suited { high: v, .. }
        | CanonicalPreflopHand::Offsuit { high: v, .. } => value_rank_index(v),
    };
    let low = match canonical {
        CanonicalPreflopHand::Pair(v) => value_rank_index(v),
        CanonicalPreflopHand::Suited { low, .. } | CanonicalPreflopHand::Offsuit { low, .. } => {
            value_rank_index(low)
        }
    };
    match canonical {
        CanonicalPreflopHand::Pair(_) => high * 13 + low,
        CanonicalPreflopHand::Suited { .. } => high * 13 + low,
        CanonicalPreflopHand::Offsuit { .. } => low * 13 + high,
    }
}

fn initialize_centroids(data: &[Vec<f64>], k: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.shuffle(rng);
    indices
        .into_iter()
        .take(k)
        .map(|idx| normalize_histogram(&data[idx]))
        .collect()
}

fn nearest_centroid(point: &[f64], centroids: &[Vec<f64>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(idx, centroid)| (idx, emd_distance(point, centroid)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        .map(|(idx, _)| idx)
        .expect("at least one centroid is required")
}

fn mean_point(group: &[&[f64]], dims: usize) -> Vec<f64> {
    let mut acc = vec![0.0; dims];
    for point in group {
        for (idx, value) in point.iter().copied().enumerate() {
            acc[idx] += value;
        }
    }
    let denom = group.len() as f64;
    acc.iter_mut().for_each(|x| *x /= denom);
    acc
}

fn farthest_point_index(data: &[Vec<f64>], centroids: &[Vec<f64>], assignments: &[usize]) -> usize {
    data.iter()
        .enumerate()
        .map(|(idx, point)| {
            let centroid = &centroids[assignments[idx]];
            (idx, emd_distance(point, centroid))
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn read_u32(reader: &mut File) -> io::Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_f64(reader: &mut File) -> io::Result<f64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(f64::from_le_bytes(bytes))
}

fn order_by_rank_desc(a: Card, b: Card) -> (Card, Card) {
    let a_idx = value_rank_index(a.value);
    let b_idx = value_rank_index(b.value);
    if a_idx <= b_idx {
        (a, b)
    } else {
        (b, a)
    }
}

fn value_rank_index(value: RsValue) -> usize {
    match value {
        RsValue::Ace => 0,
        RsValue::King => 1,
        RsValue::Queen => 2,
        RsValue::Jack => 3,
        RsValue::Ten => 4,
        RsValue::Nine => 5,
        RsValue::Eight => 6,
        RsValue::Seven => 7,
        RsValue::Six => 8,
        RsValue::Five => 9,
        RsValue::Four => 10,
        RsValue::Three => 11,
        RsValue::Two => 12,
    }
}

#[cfg(test)]
mod tests {
    use crate::clustering::{
        canonical_preflop_index, canonicalize_preflop_hand, emd_distance, kmeans_emd,
        load_clustering_result, save_clustering_result, BucketStreet, CanonicalPreflopHand,
        KMeansConfig,
    };
    use rs_poker::core::{Card, Suit, Value};
    use std::env;
    use std::fs;

    #[test]
    fn emd_is_zero_for_identical_histograms() {
        let a = vec![0.0, 0.2, 0.3, 0.5];
        let b = vec![0.0, 0.2, 0.3, 0.5];
        assert_eq!(emd_distance(&a, &b), 0.0);
    }

    #[test]
    fn kmeans_puts_aa_and_kk_in_same_nearby_bucket() {
        let data = vec![
            vec![0.0, 0.0, 0.0, 0.05, 0.95],    // AA
            vec![0.0, 0.0, 0.0, 0.12, 0.88],    // KK
            vec![0.45, 0.35, 0.15, 0.04, 0.01], // weak showdown profile
            vec![0.08, 0.40, 0.34, 0.16, 0.02], // draw-ish
        ];
        let result = kmeans_emd(
            &data,
            &KMeansConfig {
                k: 2,
                max_iters: 100,
                seed: 42,
            },
        )
        .expect("clustering should succeed");

        assert_eq!(
            result.assignments[0], result.assignments[1],
            "AA and KK should be nearby"
        );
    }

    #[test]
    fn kmeans_separates_flush_draw_and_gutshot_profiles() {
        let data = vec![
            vec![0.00, 0.10, 0.30, 0.40, 0.20], // flush draw: high semi-made equity
            vec![0.20, 0.45, 0.25, 0.08, 0.02], // gutshot: lower equity profile
            vec![0.00, 0.00, 0.00, 0.08, 0.92], // overpair-like
            vec![0.00, 0.00, 0.01, 0.15, 0.84], // strong made hand
            vec![0.52, 0.30, 0.13, 0.04, 0.01], // weak air-like hand
        ];
        let result = kmeans_emd(
            &data,
            &KMeansConfig {
                k: 3,
                max_iters: 100,
                seed: 99,
            },
        )
        .expect("clustering should succeed");
        let flush_cluster = result.assignments[0];
        let gutshot_cluster = result.assignments[1];
        assert_ne!(
            flush_cluster, gutshot_cluster,
            "flush draw and gutshot should separate"
        );
    }

    #[test]
    fn clustering_result_roundtrip_binary_io() {
        let data = vec![
            vec![0.0, 0.0, 0.0, 0.05, 0.95],
            vec![0.0, 0.0, 0.0, 0.12, 0.88],
            vec![0.08, 0.40, 0.34, 0.16, 0.02],
        ];
        let result = kmeans_emd(
            &data,
            &KMeansConfig {
                k: 2,
                max_iters: 50,
                seed: 9,
            },
        )
        .expect("clustering should succeed");

        let path = env::temp_dir().join("talibus_abstraction_roundtrip.clusters");
        save_clustering_result(&path, &result).expect("save should succeed");
        let loaded = load_clustering_result(&path).expect("load should succeed");
        fs::remove_file(path).ok();

        assert_eq!(loaded.assignments, result.assignments);
        assert_eq!(loaded.centroids.len(), result.centroids.len());
        assert_eq!(loaded.centroids[0].len(), result.centroids[0].len());
    }

    #[test]
    fn preflop_canonicalization_collapses_suits() {
        let hand_1 = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ];
        let hand_2 = [
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::King, Suit::Heart),
        ];
        let can_1 = canonicalize_preflop_hand(hand_1);
        let can_2 = canonicalize_preflop_hand(hand_2);
        assert_eq!(can_1, can_2);
        assert_eq!(
            can_1,
            CanonicalPreflopHand::Suited {
                high: Value::Ace,
                low: Value::King
            }
        );
    }

    #[test]
    fn canonical_index_fits_169_grid() {
        let pocket_aces = CanonicalPreflopHand::Pair(Value::Ace);
        let ak_suited = CanonicalPreflopHand::Suited {
            high: Value::Ace,
            low: Value::King,
        };
        let ak_off = CanonicalPreflopHand::Offsuit {
            high: Value::Ace,
            low: Value::King,
        };
        let aa_idx = canonical_preflop_index(pocket_aces);
        let aks_idx = canonical_preflop_index(ak_suited);
        let ako_idx = canonical_preflop_index(ak_off);

        assert!(aa_idx < 169);
        assert!(aks_idx < 169);
        assert!(ako_idx < 169);
        assert_ne!(aks_idx, ako_idx);
    }

    #[test]
    fn bucket_counts_match_plan() {
        use crate::clustering::buckets_for_street;
        assert_eq!(buckets_for_street(BucketStreet::Preflop), 169);
        assert_eq!(buckets_for_street(BucketStreet::Flop), 500);
        assert_eq!(buckets_for_street(BucketStreet::Turn), 500);
        assert_eq!(buckets_for_street(BucketStreet::River), 500);
    }
}
