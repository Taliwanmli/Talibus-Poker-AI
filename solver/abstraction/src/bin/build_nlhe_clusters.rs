use abstraction::clustering::{
    buckets_for_street, canonical_preflop_index, canonicalize_preflop_hand, kmeans_emd,
    save_clustering_result, BucketStreet, ClusteringResult, KMeansConfig,
};
use abstraction::equity::{compute_equity_multiway, equity_to_histogram, EQUITY_DIMS};
use rand::prelude::{IndexedRandom, SeedableRng, StdRng};
use rayon::prelude::*;
use rs_poker::core::{Card, Suit, Value};
use std::fs;
use std::path::{Path, PathBuf};

const POST_FLOP_OPPONENT_SAMPLES: usize = 200;
const DEFAULT_SAMPLES_PER_STREET: usize = 200_000;
const DEFAULT_MAX_ITERS: usize = 40;
const DEFAULT_NUM_PLAYERS: usize = 6;

fn flag_value<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.windows(2).find_map(|w| {
        if w[0] == name {
            Some(w[1].as_str())
        } else {
            None
        }
    })
}

fn parse_u64_arg(args: &[String], name: &str, default: u64) -> u64 {
    flag_value(args, name)
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

fn parse_usize_arg(args: &[String], name: &str, default: usize) -> usize {
    flag_value(args, name)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_path_arg(args: &[String], name: &str, default: &str) -> PathBuf {
    PathBuf::from(flag_value(args, name).unwrap_or(default))
}

fn cluster_filename(street: BucketStreet) -> &'static str {
    match street {
        BucketStreet::Preflop => "preflop.clusters",
        BucketStreet::Flop => "flop.clusters",
        BucketStreet::Turn => "turn.clusters",
        BucketStreet::River => "river.clusters",
    }
}

fn build_deck() -> Vec<Card> {
    let mut deck = Vec::with_capacity(52);
    for suit in Suit::suits() {
        for value in Value::values() {
            deck.push(Card::new(value, suit));
        }
    }
    deck
}

fn collect_preflop_canonical_coverage(deck: &[Card]) -> Vec<usize> {
    let mut canonical_counts = vec![0usize; buckets_for_street(BucketStreet::Preflop)];
    for i in 0..deck.len() {
        for j in (i + 1)..deck.len() {
            let hand = [deck[i], deck[j]];
            let canonical_idx = canonical_preflop_index(canonicalize_preflop_hand(hand));
            canonical_counts[canonical_idx] = canonical_counts[canonical_idx].saturating_add(1);
        }
    }
    canonical_counts
}

fn build_preflop_identity_clustering(deck: &[Card]) -> ClusteringResult {
    let canonical_counts = collect_preflop_canonical_coverage(deck);
    for (idx, count) in canonical_counts.iter().copied().enumerate() {
        if count == 0 {
            panic!("missing canonical preflop class coverage for index {idx}");
        }
    }

    let k = buckets_for_street(BucketStreet::Preflop);
    let centroids = (0..k)
        .map(|idx| {
            let mut centroid = vec![0.0; k];
            centroid[idx] = 1.0;
            centroid
        })
        .collect::<Vec<_>>();

    ClusteringResult {
        assignments: (0..k).collect(),
        centroids,
        iterations_run: 0,
    }
}

fn save_preflop_clustering(out_dir: &Path, deck: &[Card]) {
    let clustering = build_preflop_identity_clustering(deck);
    let out_path = out_dir.join(cluster_filename(BucketStreet::Preflop));
    save_clustering_result(&out_path, &clustering).unwrap_or_else(|err| {
        panic!(
            "failed to save {:?} to {}: {err}",
            BucketStreet::Preflop,
            out_path.display()
        )
    });
    println!(
        "wrote {:?} clusters -> {} (deterministic canonical identity map)",
        BucketStreet::Preflop,
        out_path.display()
    );
}

fn collect_postflop_dataset(
    deck: &[Card],
    board_card_count: usize,
    sample_count: usize,
    num_players: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let cards_needed = 2 + board_card_count;
    let opponent_count = num_players.saturating_sub(1).max(1);
    (0..sample_count)
        .into_par_iter()
        .map(|sample_idx| {
            let mut rng = StdRng::seed_from_u64(
                seed ^ ((sample_idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            );
            let sampled = deck
                .choose_multiple(&mut rng, cards_needed)
                .copied()
                .collect::<Vec<_>>();
            let hole = [sampled[0], sampled[1]];
            let board = &sampled[2..];
            let equity = compute_equity_multiway(
                hole,
                board,
                opponent_count,
                POST_FLOP_OPPONENT_SAMPLES,
                &mut rng,
            );
            let hist = equity_to_histogram(equity);
            debug_assert_eq!(hist.len(), EQUITY_DIMS);
            hist
        })
        .collect()
}

fn street_board_card_count(street: BucketStreet) -> usize {
    match street {
        BucketStreet::Preflop => 0,
        BucketStreet::Flop => 3,
        BucketStreet::Turn => 4,
        BucketStreet::River => 5,
    }
}

fn cluster_and_save_street(
    out_dir: &Path,
    street: BucketStreet,
    data: Vec<Vec<f64>>,
    seed: u64,
    max_iters: usize,
) {
    let k = buckets_for_street(street);
    if data.len() < k {
        panic!(
            "insufficient samples for {:?}: {} samples for {} buckets",
            street,
            data.len(),
            k
        );
    }
    println!(
        "clustering {:?}: samples={}, buckets={}, max_iters={}",
        street,
        data.len(),
        k,
        max_iters
    );
    let clustering = kmeans_emd(&data, &KMeansConfig { k, max_iters, seed })
        .unwrap_or_else(|err| panic!("failed to cluster {:?}: {:?}", street, err));

    let out_path = out_dir.join(cluster_filename(street));
    save_clustering_result(&out_path, &clustering).unwrap_or_else(|err| {
        panic!(
            "failed to save {:?} to {}: {err}",
            street,
            out_path.display()
        )
    });
    println!(
        "wrote {:?} clusters -> {} (iterations_run={})",
        street,
        out_path.display(),
        clustering.iterations_run
    );
}

fn print_help() {
    println!(
        "build_nlhe_clusters options:\n\
         --out-dir <path> (default checkpoints/nlhe_clusters)\n\
         --seed <u64> (default 17)\n\
         --samples-per-street <usize> (default 50000)\n\
         --max-iters <usize> (default 40)\n\
         --num-players <usize> (default {DEFAULT_NUM_PLAYERS})"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_help();
        return;
    }

    let out_dir = parse_path_arg(&args, "--out-dir", "checkpoints/nlhe_clusters");
    let seed = parse_u64_arg(&args, "--seed", 17);
    let samples_per_street =
        parse_usize_arg(&args, "--samples-per-street", DEFAULT_SAMPLES_PER_STREET);
    let max_iters = parse_usize_arg(&args, "--max-iters", DEFAULT_MAX_ITERS);
    let num_players = parse_usize_arg(&args, "--num-players", DEFAULT_NUM_PLAYERS).max(2);

    fs::create_dir_all(&out_dir).unwrap_or_else(|err| {
        panic!(
            "failed to create output directory {}: {err}",
            out_dir.display()
        )
    });

    println!(
        "building NLHE clusters: out_dir={}, seed={}, samples_per_street={}, max_iters={}, num_players={}",
        out_dir.display(),
        seed,
        samples_per_street,
        max_iters,
        num_players
    );

    let deck = build_deck();

    save_preflop_clustering(&out_dir, &deck);

    for (idx, street) in [BucketStreet::Flop, BucketStreet::Turn, BucketStreet::River]
        .iter()
        .copied()
        .enumerate()
    {
        let bucket_count = buckets_for_street(street);
        let sample_count = samples_per_street.max(bucket_count);
        let board_card_count = street_board_card_count(street);
        let street_seed = seed ^ ((idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        println!(
            "sampling {:?}: board_cards={}, samples={}",
            street, board_card_count, sample_count
        );
        let data = collect_postflop_dataset(
            &deck,
            board_card_count,
            sample_count,
            num_players,
            street_seed,
        );
        cluster_and_save_street(&out_dir, street, data, street_seed, max_iters);
    }
}
