use abstraction::action_abstraction::{map_real_bet_to_abstract, AbstractAction, Street};
use abstraction::clustering::{
    buckets_for_street, emd_distance, load_clustering_result, BucketStreet, ClusteringResult,
};
use player::{BlueprintPlayer, BlueprintTable, PlayerMode};
use rs_poker::core::{Card, Suit, Value as CardValue};
use serde_json::{json, Value};
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

const CLUSTER_DIMENSIONS: usize = 13;

fn env_or_default(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn cli_flag_value<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.windows(2).find_map(|w| {
        if w[0] == name {
            Some(w[1].as_str())
        } else {
            None
        }
    })
}

fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => default,
        },
        Err(_) => default,
    }
}

fn parse_mode(raw: &str) -> PlayerMode {
    match raw.trim().to_ascii_lowercase().as_str() {
        "argmax" => PlayerMode::ArgMax,
        _ => PlayerMode::Sample,
    }
}

#[derive(Clone)]
struct StreetClusters {
    centroids: Vec<Vec<f64>>,
}

#[derive(Clone)]
struct LoadedClusters {
    preflop: StreetClusters,
    flop: StreetClusters,
    turn: StreetClusters,
    river: StreetClusters,
}

impl LoadedClusters {
    fn load(cluster_dir: &Path) -> Result<Self, String> {
        Ok(Self {
            preflop: load_street_clusters(cluster_dir, BucketStreet::Preflop)?,
            flop: load_street_clusters(cluster_dir, BucketStreet::Flop)?,
            turn: load_street_clusters(cluster_dir, BucketStreet::Turn)?,
            river: load_street_clusters(cluster_dir, BucketStreet::River)?,
        })
    }

    fn for_street(&self, street: BucketStreet) -> &StreetClusters {
        match street {
            BucketStreet::Preflop => &self.preflop,
            BucketStreet::Flop => &self.flop,
            BucketStreet::Turn => &self.turn,
            BucketStreet::River => &self.river,
        }
    }
}

fn cluster_filename(street: BucketStreet) -> &'static str {
    match street {
        BucketStreet::Preflop => "preflop.clusters",
        BucketStreet::Flop => "flop.clusters",
        BucketStreet::Turn => "turn.clusters",
        BucketStreet::River => "river.clusters",
    }
}

fn load_street_clusters(
    cluster_dir: &Path,
    street: BucketStreet,
) -> Result<StreetClusters, String> {
    let path = cluster_dir.join(cluster_filename(street));
    let clustering = load_clustering_result(&path)
        .map_err(|err| format!("failed to load {}: {err}", path.display()))?;
    validate_clustering(&clustering, street, &path)?;
    Ok(StreetClusters {
        centroids: clustering.centroids,
    })
}

fn validate_clustering(
    clustering: &ClusteringResult,
    street: BucketStreet,
    path: &Path,
) -> Result<(), String> {
    if clustering.centroids.is_empty() {
        return Err(format!(
            "{:?} clustering {} has no centroids",
            street,
            path.display()
        ));
    }
    let expected_bucket_count = buckets_for_street(street);
    if clustering.centroids.len() != expected_bucket_count {
        return Err(format!(
            "{:?} clustering {} expected {} centroids, got {}",
            street,
            path.display(),
            expected_bucket_count,
            clustering.centroids.len()
        ));
    }
    for centroid in &clustering.centroids {
        if centroid.len() != CLUSTER_DIMENSIONS {
            return Err(format!(
                "{:?} clustering {} expected centroid dimension {}, got {}",
                street,
                path.display(),
                CLUSTER_DIMENSIONS,
                centroid.len()
            ));
        }
    }
    Ok(())
}

fn hash_tokens(tokens: &Vec<String>) -> u64 {
    let mut hasher = DefaultHasher::new();
    tokens.hash(&mut hasher);
    hasher.finish()
}

fn get_street(v: &Value) -> Street {
    match v
        .get("street")
        .and_then(Value::as_str)
        .unwrap_or("flop")
        .to_ascii_lowercase()
        .as_str()
    {
        "preflop" => Street::Preflop,
        "turn" => Street::Turn,
        "river" => Street::River,
        _ => Street::Flop,
    }
}

fn street_from_raw(raw: &str) -> Street {
    match raw.trim().to_ascii_lowercase().as_str() {
        "preflop" => Street::Preflop,
        "turn" => Street::Turn,
        "river" => Street::River,
        _ => Street::Flop,
    }
}

fn street_code(street: Street) -> &'static str {
    match street {
        Street::Preflop => "P",
        Street::Flop => "F",
        Street::Turn => "T",
        Street::River => "R",
    }
}

fn bucket_street(street: Street) -> BucketStreet {
    match street {
        Street::Preflop => BucketStreet::Preflop,
        Street::Flop => BucketStreet::Flop,
        Street::Turn => BucketStreet::Turn,
        Street::River => BucketStreet::River,
    }
}

fn parse_rank(raw: char) -> Option<CardValue> {
    match raw.to_ascii_uppercase() {
        'A' => Some(CardValue::Ace),
        'K' => Some(CardValue::King),
        'Q' => Some(CardValue::Queen),
        'J' => Some(CardValue::Jack),
        'T' => Some(CardValue::Ten),
        '9' => Some(CardValue::Nine),
        '8' => Some(CardValue::Eight),
        '7' => Some(CardValue::Seven),
        '6' => Some(CardValue::Six),
        '5' => Some(CardValue::Five),
        '4' => Some(CardValue::Four),
        '3' => Some(CardValue::Three),
        '2' => Some(CardValue::Two),
        _ => None,
    }
}

fn parse_suit(raw: char) -> Option<Suit> {
    match raw.to_ascii_lowercase() {
        'c' => Some(Suit::Club),
        'd' => Some(Suit::Diamond),
        'h' => Some(Suit::Heart),
        's' => Some(Suit::Spade),
        _ => None,
    }
}

fn parse_card_code(raw: &str) -> Option<Card> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    let (rank_ch, suit_ch) = if trimmed.len() == 3 && trimmed[..2].eq_ignore_ascii_case("10") {
        ('T', trimmed.chars().nth(2)?)
    } else if trimmed.len() >= 2 {
        (trimmed.chars().next()?, trimmed.chars().nth(1)?)
    } else {
        return None;
    };

    let value = parse_rank(rank_ch)?;
    let suit = parse_suit(suit_ch)?;
    Some(Card::new(value, suit))
}

fn parse_cards(raw: &Value) -> Vec<Card> {
    raw.as_array()
        .map(|cards| {
            cards
                .iter()
                .filter_map(Value::as_str)
                .filter_map(parse_card_code)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn rank_hist_index(value: CardValue) -> usize {
    match value {
        CardValue::Ace => 0,
        CardValue::King => 1,
        CardValue::Queen => 2,
        CardValue::Jack => 3,
        CardValue::Ten => 4,
        CardValue::Nine => 5,
        CardValue::Eight => 6,
        CardValue::Seven => 7,
        CardValue::Six => 8,
        CardValue::Five => 9,
        CardValue::Four => 10,
        CardValue::Three => 11,
        CardValue::Two => 12,
    }
}

fn normalized_rank_histogram(cards: &[Card]) -> Vec<f64> {
    let mut hist = vec![0.0; CLUSTER_DIMENSIONS];
    if cards.is_empty() {
        return hist;
    }
    for card in cards {
        hist[rank_hist_index(card.value)] += 1.0;
    }
    let denom = cards.len() as f64;
    hist.iter_mut().for_each(|v| *v /= denom);
    hist
}

fn assign_bucket(hist: &[f64], clusters: &StreetClusters) -> u16 {
    clusters
        .centroids
        .iter()
        .enumerate()
        .map(|(idx, centroid)| (idx, emd_distance(hist, centroid)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        .map(|(idx, _)| idx as u16)
        .unwrap_or(0)
}

fn card_bucket_for_state(
    street: Street,
    hero_cards: &[Card],
    board_cards: &[Card],
    clusters: &LoadedClusters,
) -> u16 {
    let mut cards = Vec::with_capacity(hero_cards.len() + board_cards.len());
    cards.extend_from_slice(hero_cards);
    cards.extend_from_slice(board_cards);
    let hist = normalized_rank_histogram(&cards);
    assign_bucket(&hist, clusters.for_street(bucket_street(street)))
}

fn board_bucket_for_state(street: Street, board_cards: &[Card], clusters: &LoadedClusters) -> u16 {
    if board_cards.is_empty() {
        return 0;
    }
    let hist = normalized_rank_histogram(board_cards);
    assign_bucket(&hist, clusters.for_street(bucket_street(street)))
}

fn abstract_action_token(abstract_action: AbstractAction) -> String {
    match abstract_action {
        AbstractAction::Fold => "fold".to_string(),
        AbstractAction::Check => "check".to_string(),
        AbstractAction::Call => "call".to_string(),
        AbstractAction::AllIn => "allin".to_string(),
        AbstractAction::BetPotFraction(v) => format!("bet_{v:.2}"),
        AbstractAction::RaiseMultiplier(v) => format!("raise_{v:.2}"),
        AbstractAction::OpenRaiseBbMultiple(v) => format!("open_{v:.2}"),
        AbstractAction::FourBetMultiplier(v) => format!("4bet_{v:.2}"),
    }
}

fn preflop_reraise_family(existing_tokens: &[String]) -> String {
    let prior_preflop_raises = existing_tokens
        .iter()
        .filter(|token| token.starts_with("b:open_") || token.starts_with("r:"))
        .count() as u8;
    let n_bet = prior_preflop_raises.saturating_add(2);
    format!("{n_bet}bet")
}

fn event_to_token(event: &Value, existing_tokens: &[String]) -> Option<String> {
    if let Some(raw) = event.as_str() {
        let token = raw.trim();
        return if token.is_empty() {
            None
        } else {
            Some(token.to_string())
        };
    }

    let action = event
        .get("action")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    match action.as_str() {
        "fold" => Some("f".to_string()),
        "check" => Some("x".to_string()),
        "call" => Some("c".to_string()),
        "bet" => {
            let street = event
                .get("street")
                .and_then(Value::as_str)
                .map(street_from_raw)
                .unwrap_or(Street::Flop);
            let amount = event
                .get("amountBb")
                .and_then(Value::as_f64)
                .unwrap_or(0.0)
                .max(0.0);
            let to_call_before = event
                .get("toCallBbBefore")
                .and_then(Value::as_f64)
                .unwrap_or(0.0)
                .max(0.0);
            let pot_before = (amount + to_call_before).max(1.0);
            let mapped = map_real_bet_to_abstract(street, amount.max(1.0), pot_before);
            if to_call_before > 0.0 {
                if street == Street::Preflop {
                    match mapped {
                        AbstractAction::FourBetMultiplier(v)
                        | AbstractAction::RaiseMultiplier(v) => {
                            let family = preflop_reraise_family(existing_tokens);
                            Some(format!("r:{family}_{v:.2}x"))
                        }
                        _ => Some(format!("r:{}", abstract_action_token(mapped))),
                    }
                } else {
                    Some(format!("r:{}", abstract_action_token(mapped)))
                }
            } else {
                Some(format!("b:{}", abstract_action_token(mapped)))
            }
        }
        _ => None,
    }
}

fn public_action_tokens(hand_state: &Value) -> Vec<String> {
    let events = hand_state
        .get("publicActionHistory")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut tokens = Vec::with_capacity(events.len());
    for event in events {
        if let Some(token) = event_to_token(&event, &tokens) {
            tokens.push(token);
        }
    }
    tokens
}

fn build_infoset_key(
    request: &Value,
    clusters: Option<&LoadedClusters>,
    player_index: usize,
) -> String {
    let hand_state = request.get("handState").unwrap_or(&Value::Null);
    let preflop = hand_state.get("preflop").unwrap_or(&Value::Null);
    let street = get_street(request);
    let street_state_name = match street {
        Street::Preflop => "preflop",
        Street::Flop => "flop",
        Street::Turn => "turn",
        Street::River => "river",
    };
    let street_state = hand_state.get(street_state_name).unwrap_or(&Value::Null);

    let hero_cards = parse_cards(preflop.get("heroHand").unwrap_or(&Value::Null));
    let board_cards = parse_cards(street_state.get("boardCards").unwrap_or(&Value::Null));

    let (card_bucket, board_bucket) = if let Some(loaded) = clusters {
        (
            card_bucket_for_state(street, &hero_cards, &board_cards, loaded),
            board_bucket_for_state(street, &board_cards, loaded),
        )
    } else {
        (0u16, 0u16)
    };

    let tokens = public_action_tokens(hand_state);
    let action_history_hash = hash_tokens(&tokens);

    format!(
        "p={}|street={}|cb={}|bb={}|h={}",
        player_index,
        street_code(street),
        card_bucket,
        board_bucket,
        action_history_hash
    )
}

fn to_call_bb(request: &Value) -> f64 {
    let hand_state = request.get("handState").unwrap_or(&Value::Null);
    let street = request
        .get("street")
        .and_then(Value::as_str)
        .unwrap_or("flop")
        .to_ascii_lowercase();
    hand_state
        .get(&street)
        .and_then(|street_state| street_state.get("toCallBb"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
        .max(0.0)
}

fn pot_size_bb(request: &Value) -> f64 {
    let hand_state = request.get("handState").unwrap_or(&Value::Null);
    let street = request
        .get("street")
        .and_then(Value::as_str)
        .unwrap_or("flop")
        .to_ascii_lowercase();
    hand_state
        .get(&street)
        .and_then(|street_state| street_state.get("potSizeBb"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
        .max(0.0)
}

fn legal_actions_count_for_state(request: &Value) -> usize {
    let to_call = to_call_bb(request);
    let street = get_street(request);
    if to_call > 0.0 {
        4 // fold/call/raise/all-in
    } else {
        match street {
            Street::Flop | Street::Turn => 5, // check/bet small/bet medium/bet big/all-in
            Street::Preflop | Street::River => 4,
        }
    }
}

fn action_payload_from_index(request: &Value, action_idx: usize) -> Value {
    let to_call = to_call_bb(request);
    let pot = pot_size_bb(request).max(1.0);
    let street = get_street(request);
    let is_preflop = matches!(street, Street::Preflop);

    if to_call > 0.0 {
        match action_idx {
            0 => json!({"type":"fold","note":"[blueprint] fold"}),
            1 => json!({"type":"call","note":"[blueprint] call"}),
            2 => {
                let raise_size = if is_preflop {
                    (to_call + 2.5).max(to_call + 1.0)
                } else {
                    (to_call + 0.75 * pot).max(to_call + 1.0)
                };
                json!({"type":"raise","sizeBb":raise_size,"note":"[blueprint] raise"})
            }
            _ => json!({"type":"all_in","note":"[blueprint] all_in"}),
        }
    } else {
        match street {
            Street::Preflop => match action_idx {
                0 => json!({"type":"check","note":"[blueprint] check"}),
                1 => json!({"type":"raise","sizeBb":2.5,"note":"[blueprint] open_2.50"}),
                2 => json!({"type":"raise","sizeBb":3.0,"note":"[blueprint] open_3.00"}),
                _ => json!({"type":"all_in","note":"[blueprint] all_in"}),
            },
            Street::Flop => match action_idx {
                0 => json!({"type":"check","note":"[blueprint] check"}),
                1 => json!({"type":"bet","sizeBb":0.33 * pot,"note":"[blueprint] bet_0.33"}),
                2 => json!({"type":"bet","sizeBb":0.67 * pot,"note":"[blueprint] bet_0.67"}),
                3 => json!({"type":"bet","sizeBb":1.25 * pot,"note":"[blueprint] bet_1.25"}),
                _ => json!({"type":"all_in","note":"[blueprint] all_in"}),
            },
            Street::Turn => match action_idx {
                0 => json!({"type":"check","note":"[blueprint] check"}),
                1 => json!({"type":"bet","sizeBb":0.50 * pot,"note":"[blueprint] bet_0.50"}),
                2 => json!({"type":"bet","sizeBb":0.75 * pot,"note":"[blueprint] bet_0.75"}),
                3 => json!({"type":"bet","sizeBb":1.25 * pot,"note":"[blueprint] bet_1.25"}),
                _ => json!({"type":"all_in","note":"[blueprint] all_in"}),
            },
            Street::River => match action_idx {
                0 => json!({"type":"check","note":"[blueprint] check"}),
                1 => json!({"type":"bet","sizeBb":0.50 * pot,"note":"[blueprint] bet_0.50"}),
                2 => json!({"type":"bet","sizeBb":1.00 * pot,"note":"[blueprint] bet_1.00"}),
                _ => json!({"type":"all_in","note":"[blueprint] all_in"}),
            },
        }
    }
}

fn print_help() {
    println!(
        "blueprint_policy_worker options:\n\
         --cluster-dir <path> (default from WIPOKER_CLUSTER_DIR or checkpoints/nlhe_clusters)\n\
         environment:\n\
         WIPOKER_BLUEPRINT_FILE (default checkpoints/latest.blueprint)\n\
         WIPOKER_BLUEPRINT_MODE sample|argmax (default sample)\n\
         WIPOKER_BLUEPRINT_SEED (default 17)\n\
         WIPOKER_BLUEPRINT_PLAYER_INDEX (default 0)\n\
         WIPOKER_BLUEPRINT_STRICT true|false (default false)"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_help();
        return;
    }

    let blueprint_file = env_or_default("WIPOKER_BLUEPRINT_FILE", "checkpoints/latest.blueprint");
    let mode = parse_mode(&env_or_default("WIPOKER_BLUEPRINT_MODE", "sample"));
    let seed_base = env_or_default("WIPOKER_BLUEPRINT_SEED", "17")
        .parse::<u64>()
        .unwrap_or(17);
    let player_index = env_or_default("WIPOKER_BLUEPRINT_PLAYER_INDEX", "0")
        .parse::<usize>()
        .unwrap_or(0);
    let strict_blueprint_loading = env_flag("WIPOKER_BLUEPRINT_STRICT", false);
    let cluster_dir_raw = cli_flag_value(&args, "--cluster-dir")
        .map(|v| v.to_string())
        .unwrap_or_else(|| env_or_default("WIPOKER_CLUSTER_DIR", "checkpoints/nlhe_clusters"));
    let cluster_dir = PathBuf::from(cluster_dir_raw);

    let blueprint_path = PathBuf::from(&blueprint_file);
    let table = match BlueprintTable::load_from_file(&blueprint_path) {
        Ok(t) => t,
        Err(err) => {
            if strict_blueprint_loading {
                let _ = writeln!(
                    io::stderr(),
                    "[blueprint_policy_worker] failed to load blueprint {}: {}",
                    blueprint_path.display(),
                    err
                );
                std::process::exit(1);
            }
            let _ = writeln!(
                io::stderr(),
                "[blueprint_policy_worker] warning: failed to load blueprint {}; using empty fallback table ({})",
                blueprint_path.display(),
                err
            );
            BlueprintTable::default()
        }
    };
    let player = BlueprintPlayer::new(table, mode);
    let clusters = match LoadedClusters::load(&cluster_dir) {
        Ok(loaded) => Some(loaded),
        Err(err) => {
            if strict_blueprint_loading {
                let _ = writeln!(
                    io::stderr(),
                    "[blueprint_policy_worker] failed to load clusters from {}: {}",
                    cluster_dir.display(),
                    err
                );
                std::process::exit(1);
            }
            let _ = writeln!(
                io::stderr(),
                "[blueprint_policy_worker] warning: failed to load clusters from {}; using zero buckets fallback ({})",
                cluster_dir.display(),
                err
            );
            None
        }
    };

    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    for (line_idx, line_result) in stdin.lock().lines().enumerate() {
        let line = match line_result {
            Ok(l) => l,
            Err(err) => {
                let payload = json!({
                    "status":"unavailable",
                    "recommendedAction":Value::Null,
                    "mix":Value::Null,
                    "error": format!("read_error: {}", err),
                });
                let _ = writeln!(stdout, "{}", payload);
                continue;
            }
        };
        if line.trim().is_empty() {
            continue;
        }

        let request: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(err) => {
                let payload = json!({
                    "status":"unavailable",
                    "recommendedAction":Value::Null,
                    "mix":Value::Null,
                    "error": format!("invalid_json: {}", err),
                });
                let _ = writeln!(stdout, "{}", payload);
                continue;
            }
        };

        let infoset_key = build_infoset_key(&request, clusters.as_ref(), player_index);
        let legal_count = legal_actions_count_for_state(&request);
        let chosen_idx = player.choose_action_index_for_key(
            &infoset_key,
            legal_count,
            seed_base ^ line_idx as u64,
        );
        let action = action_payload_from_index(&request, chosen_idx);

        let response = json!({
            "status":"ok",
            "recommendedAction": action.clone(),
            "executedAction": action.clone(),
            "argmaxAction": action,
            "mix": Value::Null,
            "debug":{
                "infoset_key": infoset_key,
                "chosen_action_index": chosen_idx,
                "legal_actions_count": legal_count,
                "clusters_loaded": clusters.is_some()
            }
        });
        let _ = writeln!(stdout, "{}", response);
    }
}
