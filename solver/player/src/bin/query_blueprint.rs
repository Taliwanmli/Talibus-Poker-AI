use player::{BlueprintInfoSetKey, BlueprintPlayer, BlueprintTable, PlayerMode};
use std::path::PathBuf;

fn parse_u16_arg(args: &[String], idx: usize, default: u16) -> u16 {
    args.get(idx)
        .and_then(|v| v.parse::<u16>().ok())
        .unwrap_or(default)
}

fn parse_u64_arg(args: &[String], idx: usize, default: u64) -> u64 {
    args.get(idx)
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

fn parse_usize_arg(args: &[String], idx: usize, default: usize) -> usize {
    args.get(idx)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_mode(args: &[String], idx: usize) -> PlayerMode {
    match args.get(idx).map(|s| s.as_str()) {
        Some("argmax") => PlayerMode::ArgMax,
        _ => PlayerMode::Sample,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: query_blueprint <blueprint.bin> [card_bucket] [board_bucket] [history_hash] [legal_count] [seed] [mode=sample|argmax]"
        );
        std::process::exit(2);
    }

    let blueprint_path = PathBuf::from(&args[1]);
    let card_bucket = parse_u16_arg(&args, 2, 0);
    let board_bucket = parse_u16_arg(&args, 3, 0);
    let history_hash = parse_u64_arg(&args, 4, 0);
    let legal_count = parse_usize_arg(&args, 5, 3);
    let seed = parse_u64_arg(&args, 6, 17);
    let mode = parse_mode(&args, 7);

    let table = match BlueprintTable::load_from_file(&blueprint_path) {
        Ok(t) => t,
        Err(err) => {
            eprintln!(
                "failed to load blueprint {}: {err}",
                blueprint_path.display()
            );
            std::process::exit(1);
        }
    };
    let infoset = BlueprintInfoSetKey {
        card_bucket,
        board_bucket,
        action_history_hash: history_hash,
    };
    let player = BlueprintPlayer::new(table, mode);
    let chosen = player.choose_action_index(&infoset, legal_count, seed);

    println!("chosen_action_index={chosen}");
    println!("infoset_key={}", infoset.encode());
    println!("legal_count={legal_count}");
}
