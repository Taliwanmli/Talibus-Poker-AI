use cfr::external_sampling::ExternalSamplingTrainer;
use player::{BlueprintPolicy, BlueprintTable};
use std::collections::HashMap;
use std::path::PathBuf;

fn flag_value<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.windows(2).find_map(|w| {
        if w[0] == name {
            Some(w[1].as_str())
        } else {
            None
        }
    })
}

fn parse_path_arg(args: &[String], name: &str, default: &str) -> PathBuf {
    PathBuf::from(flag_value(args, name).unwrap_or(default))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!(
            "export_blueprint_from_checkpoint options:\n\
             --checkpoint <path> (default checkpoints/nlhe_hu/latest.ckpt)\n\
             --out <path> (default checkpoints/latest.blueprint)"
        );
        return;
    }

    let checkpoint_path = parse_path_arg(&args, "--checkpoint", "checkpoints/nlhe_hu/latest.ckpt");
    let out_path = parse_path_arg(&args, "--out", "checkpoints/latest.blueprint");

    let trainer =
        ExternalSamplingTrainer::load_checkpoint(&checkpoint_path).unwrap_or_else(|err| {
            panic!(
                "failed to load checkpoint {}: {err}",
                checkpoint_path.display()
            )
        });
    let avg = trainer.average_policy_table();
    let mut policies = HashMap::with_capacity(avg.len());
    for (key, action_probabilities) in avg {
        policies.insert(
            key,
            BlueprintPolicy {
                action_probabilities,
            },
        );
    }

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).unwrap_or_else(|err| {
            panic!(
                "failed to create output directory {}: {err}",
                parent.display()
            )
        });
    }
    let table = BlueprintTable { policies };
    table
        .save_to_file(&out_path)
        .unwrap_or_else(|err| panic!("failed to write blueprint {}: {err}", out_path.display()));

    println!(
        "checkpoint={} infosets={} blueprint={}",
        checkpoint_path.display(),
        table.policies.len(),
        out_path.display()
    );
}
