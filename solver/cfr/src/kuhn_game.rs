use crate::external_sampling::{GameModel, NodeKind};

#[derive(Clone)]
pub struct KuhnState {
    pub cards: Option<[u8; 2]>,
    pub history: String,
}

pub struct KuhnGame;

impl KuhnGame {
    pub const DEALS: [[u8; 2]; 6] = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]];
    pub const TARGET_EV_PLAYER0: f64 = -1.0 / 18.0;
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

pub fn terminal_utility_player0(cards: [u8; 2], history: &str) -> Option<f64> {
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
