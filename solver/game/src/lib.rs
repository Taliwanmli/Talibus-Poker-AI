use rand::prelude::{SeedableRng, SliceRandom, StdRng};
use rs_poker::core::{
    Card as RsCard, FlatHand, Rank as RsRank, Rankable, Suit as RsSuit, Value as RsValue,
};
use std::collections::HashSet;
use std::fmt::{Display, Formatter};

pub type Chips = u32;
pub const MAX_PLAYERS: usize = 6;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Suit {
    Clubs,
    Diamonds,
    Hearts,
    Spades,
}

impl Display for Suit {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let symbol = match self {
            Suit::Clubs => "c",
            Suit::Diamonds => "d",
            Suit::Hearts => "h",
            Suit::Spades => "s",
        };
        write!(f, "{symbol}")
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Rank {
    Two = 2,
    Three = 3,
    Four = 4,
    Five = 5,
    Six = 6,
    Seven = 7,
    Eight = 8,
    Nine = 9,
    Ten = 10,
    Jack = 11,
    Queen = 12,
    King = 13,
    Ace = 14,
}

impl Display for Rank {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let value = match self {
            Rank::Two => "2",
            Rank::Three => "3",
            Rank::Four => "4",
            Rank::Five => "5",
            Rank::Six => "6",
            Rank::Seven => "7",
            Rank::Eight => "8",
            Rank::Nine => "9",
            Rank::Ten => "T",
            Rank::Jack => "J",
            Rank::Queen => "Q",
            Rank::King => "K",
            Rank::Ace => "A",
        };
        write!(f, "{value}")
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Card {
    pub rank: Rank,
    pub suit: Suit,
}

impl Card {
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        if s.len() < 2 {
            return None;
        }
        let (rank_part, suit_char) = s.split_at(s.len() - 1);
        let rank = match rank_part {
            "2" => Rank::Two,
            "3" => Rank::Three,
            "4" => Rank::Four,
            "5" => Rank::Five,
            "6" => Rank::Six,
            "7" => Rank::Seven,
            "8" => Rank::Eight,
            "9" => Rank::Nine,
            "T" | "10" => Rank::Ten,
            "J" => Rank::Jack,
            "Q" => Rank::Queen,
            "K" => Rank::King,
            "A" => Rank::Ace,
            _ => return None,
        };
        let suit = match suit_char.to_ascii_lowercase().as_str() {
            "c" => Suit::Clubs,
            "d" => Suit::Diamonds,
            "h" => Suit::Hearts,
            "s" => Suit::Spades,
            _ => return None,
        };
        Some(Card { rank, suit })
    }
}

impl Display for Card {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.rank, self.suit)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SeatPosition {
    SmallBlind,
    BigBlind,
    UnderTheGun,
    Hijack,
    Cutoff,
    Button,
}

impl SeatPosition {
    pub const SIX_MAX_ORDER: [SeatPosition; MAX_PLAYERS] = [
        SeatPosition::SmallBlind,
        SeatPosition::BigBlind,
        SeatPosition::UnderTheGun,
        SeatPosition::Hijack,
        SeatPosition::Cutoff,
        SeatPosition::Button,
    ];
}

#[derive(Clone, Debug)]
pub struct Deck {
    cards: Vec<Card>,
    next_idx: usize,
}

impl Deck {
    pub fn new_ordered() -> Self {
        let mut cards = Vec::with_capacity(52);
        let suits = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];
        let ranks = [
            Rank::Two,
            Rank::Three,
            Rank::Four,
            Rank::Five,
            Rank::Six,
            Rank::Seven,
            Rank::Eight,
            Rank::Nine,
            Rank::Ten,
            Rank::Jack,
            Rank::Queen,
            Rank::King,
            Rank::Ace,
        ];

        for &rank in &ranks {
            for &suit in &suits {
                cards.push(Card { rank, suit });
            }
        }

        Self { cards, next_idx: 0 }
    }

    pub fn shuffled(seed: u64) -> Self {
        let mut deck = Self::new_ordered();
        let mut rng = StdRng::seed_from_u64(seed);
        deck.cards.shuffle(&mut rng);
        deck
    }

    pub fn from_arranged(cards: Vec<Card>) -> Self {
        Self { cards, next_idx: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.cards.len().saturating_sub(self.next_idx)
    }

    pub fn draw(&mut self) -> Option<Card> {
        if self.next_idx >= self.cards.len() {
            return None;
        }
        let card = self.cards[self.next_idx];
        self.next_idx += 1;
        Some(card)
    }
}

pub fn deck_contains_unique_cards(deck: &Deck) -> bool {
    let mut seen = HashSet::with_capacity(deck.cards.len());
    deck.cards.iter().all(|card| seen.insert(*card))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BettingRound {
    Preflop,
    Flop,
    Turn,
    River,
    Complete,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Action {
    Fold,
    Check,
    Call,
    Bet(Chips),
    RaiseTo(Chips),
    AllIn,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LegalAction {
    Fold,
    Check,
    Call { amount: Chips },
    Bet { min: Chips, max: Chips },
    RaiseTo { min: Chips, max: Chips },
    AllIn { total: Chips },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GameError {
    DeckExhausted,
    NoActorToAct,
    ActionByNonActor,
    InvalidAction,
    InvalidConfig,
}

#[derive(Clone, Copy, Debug)]
pub struct NlheConfig {
    pub num_players: usize,
    pub starting_stack: Chips,
    pub small_blind: Chips,
    pub big_blind: Chips,
}

impl Default for NlheConfig {
    fn default() -> Self {
        Self {
            num_players: 6,
            starting_stack: 10_000,
            small_blind: 50,
            big_blind: 100,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlayerState {
    pub seat: SeatPosition,
    pub stack: Chips,
    pub hole_cards: [Card; 2],
    pub folded: bool,
    pub all_in: bool,
    pub total_contribution: Chips,
    pub street_contribution: Chips,
}

#[derive(Clone, Debug)]
pub struct HandSummary {
    pub total_pot: Chips,
    pub payouts: Vec<Chips>,
    pub showdown: bool,
    pub winning_players: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct NlheGame {
    pub config: NlheConfig,
    pub round: BettingRound,
    pub players: Vec<PlayerState>,
    pub board: Vec<Card>,
    button_index: usize,
    deck: Deck,
    needs_action: Vec<bool>,
    current_actor: Option<usize>,
    current_bet: Chips,
    last_raise_size: Chips,
    summary: Option<HandSummary>,
    preflop_raise_count: u8,
}

impl NlheGame {
    pub fn new(config: NlheConfig, seed: u64) -> Result<Self, GameError> {
        if !(2..=MAX_PLAYERS).contains(&config.num_players) {
            return Err(GameError::InvalidConfig);
        }
        if config.small_blind == 0
            || config.big_blind == 0
            || config.small_blind >= config.big_blind
        {
            return Err(GameError::InvalidConfig);
        }
        if config.starting_stack < config.big_blind {
            return Err(GameError::InvalidConfig);
        }

        let mut deck = Deck::shuffled(seed);
        let mut players = Vec::with_capacity(config.num_players);
        for &seat in SeatPosition::SIX_MAX_ORDER.iter().take(config.num_players) {
            let c1 = deck.draw().ok_or(GameError::DeckExhausted)?;
            let c2 = deck.draw().ok_or(GameError::DeckExhausted)?;
            players.push(PlayerState {
                seat,
                stack: config.starting_stack,
                hole_cards: [c1, c2],
                folded: false,
                all_in: false,
                total_contribution: 0,
                street_contribution: 0,
            });
        }

        let mut game = Self {
            config,
            round: BettingRound::Preflop,
            players,
            board: Vec::with_capacity(5),
            button_index: if config.num_players == 2 {
                // In HU, SB is also the button.
                0
            } else {
                config.num_players - 1
            },
            deck,
            needs_action: vec![false; config.num_players],
            current_actor: None,
            current_bet: 0,
            last_raise_size: config.big_blind,
            summary: None,
            preflop_raise_count: 0,
        };

        game.post_forced_blinds();
        game.initialize_preflop_action_state();
        Ok(game)
    }

    pub fn new_with_deck(config: NlheConfig, deck: Deck) -> Result<Self, GameError> {
        if !(2..=MAX_PLAYERS).contains(&config.num_players) {
            return Err(GameError::InvalidConfig);
        }
        if config.small_blind == 0
            || config.big_blind == 0
            || config.small_blind >= config.big_blind
        {
            return Err(GameError::InvalidConfig);
        }
        if config.starting_stack < config.big_blind {
            return Err(GameError::InvalidConfig);
        }

        let mut deck = deck;
        let mut players = Vec::with_capacity(config.num_players);
        for &seat in SeatPosition::SIX_MAX_ORDER.iter().take(config.num_players) {
            let c1 = deck.draw().ok_or(GameError::DeckExhausted)?;
            let c2 = deck.draw().ok_or(GameError::DeckExhausted)?;
            players.push(PlayerState {
                seat,
                stack: config.starting_stack,
                hole_cards: [c1, c2],
                folded: false,
                all_in: false,
                total_contribution: 0,
                street_contribution: 0,
            });
        }

        let mut game = Self {
            config,
            round: BettingRound::Preflop,
            players,
            board: Vec::with_capacity(5),
            button_index: if config.num_players == 2 { 0 } else { config.num_players - 1 },
            deck,
            needs_action: vec![false; config.num_players],
            current_actor: None,
            current_bet: 0,
            last_raise_size: config.big_blind,
            summary: None,
            preflop_raise_count: 0,
        };

        game.post_forced_blinds();
        game.initialize_preflop_action_state();
        Ok(game)
    }

    pub fn current_actor(&self) -> Option<usize> {
        self.current_actor
    }

    pub fn summary(&self) -> Option<&HandSummary> {
        self.summary.as_ref()
    }

    pub fn is_complete(&self) -> bool {
        self.round == BettingRound::Complete
    }

    pub fn total_chips(&self) -> Chips {
        self.players.iter().map(|p| p.stack).sum()
    }

    pub fn legal_actions_for_current_actor(&self) -> Result<Vec<LegalAction>, GameError> {
        let actor = self.current_actor.ok_or(GameError::NoActorToAct)?;
        self.legal_actions(actor)
    }

    pub fn legal_actions(&self, player_idx: usize) -> Result<Vec<LegalAction>, GameError> {
        if player_idx >= self.players.len() {
            return Err(GameError::InvalidAction);
        }
        let player = &self.players[player_idx];
        if player.folded || player.all_in {
            return Ok(Vec::new());
        }

        let to_call = self.current_bet.saturating_sub(player.street_contribution);
        let max_total = player.street_contribution + player.stack;
        let mut actions = Vec::with_capacity(6);
        let call_amount = to_call.min(player.stack);
        let mut all_in_is_distinct = !(to_call > 0 && call_amount == player.stack);

        if to_call > 0 {
            actions.push(LegalAction::Fold);
            actions.push(LegalAction::Call {
                amount: call_amount,
            });
        } else {
            actions.push(LegalAction::Check);
        }

        if player.stack > 0 {
            if self.current_bet == 0 {
                if max_total >= self.config.big_blind {
                    actions.push(LegalAction::Bet {
                        min: self.config.big_blind,
                        max: max_total,
                    });
                    // Bet(max_total) is equivalent to all-in.
                    all_in_is_distinct = false;
                }
            } else if max_total > self.current_bet {
                let min_raise_to = self.current_bet.saturating_add(self.last_raise_size);
                if min_raise_to <= max_total {
                    actions.push(LegalAction::RaiseTo {
                        min: min_raise_to,
                        max: max_total,
                    });
                    // RaiseTo(max_total) is equivalent to all-in.
                    all_in_is_distinct = false;
                }
            }
            if all_in_is_distinct {
                actions.push(LegalAction::AllIn { total: max_total });
            }
        }

        Ok(actions)
    }

    pub fn preflop_raise_count(&self) -> u8 {
        self.preflop_raise_count
    }

    pub fn apply_action(&mut self, player_idx: usize, action: Action) -> Result<(), GameError> {
        if self.round == BettingRound::Complete {
            return Err(GameError::InvalidAction);
        }
        let actor = self.current_actor.ok_or(GameError::NoActorToAct)?;
        if actor != player_idx {
            return Err(GameError::ActionByNonActor);
        }
        if !self.needs_action[player_idx] {
            return Err(GameError::InvalidAction);
        }
        if self.players[player_idx].folded || self.players[player_idx].all_in {
            return Err(GameError::InvalidAction);
        }

        let to_call = self
            .current_bet
            .saturating_sub(self.players[player_idx].street_contribution);
        let max_total =
            self.players[player_idx].street_contribution + self.players[player_idx].stack;

        match action {
            Action::Fold => {
                if to_call == 0 {
                    return Err(GameError::InvalidAction);
                }
                self.players[player_idx].folded = true;
            }
            Action::Check => {
                if to_call != 0 {
                    return Err(GameError::InvalidAction);
                }
            }
            Action::Call => {
                if to_call == 0 {
                    return Err(GameError::InvalidAction);
                }
                self.commit_player_chips(player_idx, to_call);
            }
            Action::Bet(total) => {
                if self.current_bet != 0 || to_call != 0 {
                    return Err(GameError::InvalidAction);
                }
                if total < self.config.big_blind || total > max_total {
                    return Err(GameError::InvalidAction);
                }
                let delta = total.saturating_sub(self.players[player_idx].street_contribution);
                self.commit_player_chips(player_idx, delta);
                self.register_bet_increase(player_idx, total, true);
            }
            Action::RaiseTo(total) => {
                if self.current_bet == 0 || total <= self.current_bet || total > max_total {
                    return Err(GameError::InvalidAction);
                }
                let min_raise_to = self.current_bet.saturating_add(self.last_raise_size);
                let is_all_in = total == max_total;
                if total < min_raise_to && !is_all_in {
                    return Err(GameError::InvalidAction);
                }
                let delta = total.saturating_sub(self.players[player_idx].street_contribution);
                self.commit_player_chips(player_idx, delta);
                self.register_bet_increase(player_idx, total, total >= min_raise_to);
            }
            Action::AllIn => {
                if self.players[player_idx].stack == 0 {
                    return Err(GameError::InvalidAction);
                }
                let new_total = max_total;
                let delta = self.players[player_idx].stack;
                self.commit_player_chips(player_idx, delta);
                if new_total > self.current_bet {
                    let min_raise_to = self.current_bet.saturating_add(self.last_raise_size);
                    let full_raise = self.current_bet == 0 || new_total >= min_raise_to;
                    self.register_bet_increase(player_idx, new_total, full_raise);
                }
            }
        }

        self.needs_action[player_idx] = false;

        if self.active_player_count() <= 1 {
            self.resolve_uncontested();
            return Ok(());
        }

        if self.actionable_player_count() == 0 {
            self.runout_and_showdown()?;
            return Ok(());
        }

        if self.is_betting_round_complete() {
            self.advance_round()?;
            return Ok(());
        }

        self.current_actor = self.find_next_needing_action(player_idx);
        Ok(())
    }

    fn post_forced_blinds(&mut self) {
        self.commit_player_chips(0, self.config.small_blind);
        self.commit_player_chips(1, self.config.big_blind);
        self.current_bet = self.players[1].street_contribution;
        self.last_raise_size = self.config.big_blind;
    }

    fn initialize_preflop_action_state(&mut self) {
        self.needs_action = self
            .players
            .iter()
            .map(|p| !p.folded && !p.all_in)
            .collect::<Vec<_>>();

        let first_actor = self.first_preflop_actor_idx();
        if self.needs_action[first_actor] {
            self.current_actor = Some(first_actor);
        } else {
            self.current_actor = self.find_next_needing_action(first_actor);
        }
    }

    fn commit_player_chips(&mut self, player_idx: usize, requested: Chips) -> Chips {
        let player = &mut self.players[player_idx];
        let committed = requested.min(player.stack);
        player.stack -= committed;
        player.street_contribution += committed;
        player.total_contribution += committed;
        if player.stack == 0 {
            player.all_in = true;
        }
        committed
    }

    fn register_bet_increase(&mut self, actor: usize, new_total: Chips, full_raise: bool) {
        let raise_size = new_total.saturating_sub(self.current_bet);
        if self.round == BettingRound::Preflop {
            self.preflop_raise_count = self.preflop_raise_count.saturating_add(1);
        }
        self.current_bet = new_total;
        if full_raise {
            self.last_raise_size = raise_size.max(self.config.big_blind);
        }
        for idx in 0..self.players.len() {
            if idx == actor {
                continue;
            }
            if !self.players[idx].folded && !self.players[idx].all_in {
                self.needs_action[idx] = true;
            }
        }
    }

    fn first_preflop_actor_idx(&self) -> usize {
        if self.players.len() == 2 {
            0
        } else {
            2
        }
    }

    fn first_postflop_actor_idx(&self) -> usize {
        (self.button_index + 1) % self.players.len()
    }

    fn find_next_needing_action(&self, from_idx: usize) -> Option<usize> {
        let n = self.players.len();
        for offset in 1..=n {
            let idx = (from_idx + offset) % n;
            if self.needs_action[idx] && !self.players[idx].folded && !self.players[idx].all_in {
                return Some(idx);
            }
        }
        None
    }

    fn active_player_count(&self) -> usize {
        self.players.iter().filter(|p| !p.folded).count()
    }

    fn actionable_player_count(&self) -> usize {
        self.players
            .iter()
            .filter(|p| !p.folded && !p.all_in)
            .count()
    }

    fn is_betting_round_complete(&self) -> bool {
        self.players
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.folded && !p.all_in)
            .all(|(idx, _)| !self.needs_action[idx])
    }

    fn advance_round(&mut self) -> Result<(), GameError> {
        match self.round {
            BettingRound::Preflop => {
                self.burn_one()?;
                self.deal_board(3)?;
                self.round = BettingRound::Flop;
                self.prepare_new_street();
            }
            BettingRound::Flop => {
                self.burn_one()?;
                self.deal_board(1)?;
                self.round = BettingRound::Turn;
                self.prepare_new_street();
            }
            BettingRound::Turn => {
                self.burn_one()?;
                self.deal_board(1)?;
                self.round = BettingRound::River;
                self.prepare_new_street();
            }
            BettingRound::River => {
                self.resolve_showdown();
            }
            BettingRound::Complete => return Ok(()),
        }

        if self.round != BettingRound::Complete && self.actionable_player_count() == 0 {
            self.runout_and_showdown()?;
        }
        Ok(())
    }

    fn prepare_new_street(&mut self) {
        for player in &mut self.players {
            player.street_contribution = 0;
        }
        self.current_bet = 0;
        self.last_raise_size = self.config.big_blind;
        self.needs_action = self
            .players
            .iter()
            .map(|p| !p.folded && !p.all_in)
            .collect::<Vec<_>>();

        let start = self.first_postflop_actor_idx();
        self.current_actor = if self.needs_action[start] {
            Some(start)
        } else {
            self.find_next_needing_action(start)
        };
    }

    fn burn_one(&mut self) -> Result<(), GameError> {
        self.deck.draw().ok_or(GameError::DeckExhausted)?;
        Ok(())
    }

    fn deal_board(&mut self, n: usize) -> Result<(), GameError> {
        for _ in 0..n {
            self.board
                .push(self.deck.draw().ok_or(GameError::DeckExhausted)?);
        }
        Ok(())
    }

    fn runout_and_showdown(&mut self) -> Result<(), GameError> {
        while self.round != BettingRound::River {
            self.advance_round()?;
            if self.round == BettingRound::Complete {
                return Ok(());
            }
        }
        if self.round != BettingRound::Complete {
            self.resolve_showdown();
        }
        Ok(())
    }

    fn resolve_uncontested(&mut self) {
        let winner = self
            .players
            .iter()
            .enumerate()
            .find_map(|(idx, p)| if !p.folded { Some(idx) } else { None })
            .expect("must have at least one non-folded player");
        let total_pot: Chips = self.players.iter().map(|p| p.total_contribution).sum();
        self.players[winner].stack += total_pot;

        let mut payouts = vec![0; self.players.len()];
        payouts[winner] = total_pot;
        self.summary = Some(HandSummary {
            total_pot,
            payouts,
            showdown: false,
            winning_players: vec![winner],
        });
        self.round = BettingRound::Complete;
        self.current_actor = None;
        self.needs_action.fill(false);
    }

    fn resolve_showdown(&mut self) {
        if self.board.len() < 5 {
            return;
        }

        let mut ranks: Vec<Option<RsRank>> = vec![None; self.players.len()];
        for (idx, player) in self.players.iter().enumerate() {
            if player.folded {
                continue;
            }
            let mut cards = Vec::with_capacity(7);
            cards.push(to_rs_card(player.hole_cards[0]));
            cards.push(to_rs_card(player.hole_cards[1]));
            cards.extend(self.board.iter().copied().map(to_rs_card));
            ranks[idx] = Some(FlatHand::new_with_cards(cards).rank());
        }

        let contributions: Vec<Chips> = self.players.iter().map(|p| p.total_contribution).collect();
        let total_pot: Chips = contributions.iter().sum();
        let payouts = compute_side_pot_payouts(&self.players, &contributions, &ranks);

        for (idx, payout) in payouts.iter().copied().enumerate() {
            self.players[idx].stack += payout;
        }

        let max_payout = payouts.iter().copied().max().unwrap_or(0);
        let winning_players = payouts
            .iter()
            .enumerate()
            .filter_map(|(idx, &amt)| {
                if amt == max_payout && amt > 0 {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        self.summary = Some(HandSummary {
            total_pot,
            payouts,
            showdown: true,
            winning_players,
        });
        self.round = BettingRound::Complete;
        self.current_actor = None;
        self.needs_action.fill(false);
    }
}

fn compute_side_pot_payouts(
    players: &[PlayerState],
    contributions: &[Chips],
    ranks: &[Option<RsRank>],
) -> Vec<Chips> {
    let mut payouts = vec![0; players.len()];

    let mut levels = contributions
        .iter()
        .copied()
        .filter(|&value| value > 0)
        .collect::<Vec<_>>();
    levels.sort_unstable();
    levels.dedup();

    let mut previous = 0u32;
    for level in levels {
        let layer = level - previous;
        let contributors = (0..players.len())
            .filter(|&idx| contributions[idx] >= level)
            .collect::<Vec<_>>();
        let layer_pot = layer as u64 * contributors.len() as u64;

        let eligible = contributors
            .iter()
            .copied()
            .filter(|&idx| !players[idx].folded)
            .collect::<Vec<_>>();
        if eligible.is_empty() {
            previous = level;
            continue;
        }

        let best_rank = eligible
            .iter()
            .filter_map(|&idx| ranks[idx])
            .max()
            .expect("eligible showdown player must have rank");
        let mut winners = eligible
            .iter()
            .copied()
            .filter(|&idx| ranks[idx] == Some(best_rank))
            .collect::<Vec<_>>();
        winners.sort_unstable();

        let split = layer_pot / winners.len() as u64;
        let remainder = layer_pot % winners.len() as u64;
        for &winner in &winners {
            payouts[winner] += split as Chips;
        }
        for winner in winners.into_iter().take(remainder as usize) {
            payouts[winner] += 1;
        }

        previous = level;
    }

    payouts
}

fn to_rs_card(card: Card) -> RsCard {
    let value = match card.rank {
        Rank::Two => RsValue::Two,
        Rank::Three => RsValue::Three,
        Rank::Four => RsValue::Four,
        Rank::Five => RsValue::Five,
        Rank::Six => RsValue::Six,
        Rank::Seven => RsValue::Seven,
        Rank::Eight => RsValue::Eight,
        Rank::Nine => RsValue::Nine,
        Rank::Ten => RsValue::Ten,
        Rank::Jack => RsValue::Jack,
        Rank::Queen => RsValue::Queen,
        Rank::King => RsValue::King,
        Rank::Ace => RsValue::Ace,
    };
    let suit = match card.suit {
        Suit::Clubs => RsSuit::Club,
        Suit::Diamonds => RsSuit::Diamond,
        Suit::Hearts => RsSuit::Heart,
        Suit::Spades => RsSuit::Spade,
    };
    RsCard::new(value, suit)
}

#[cfg(test)]
mod tests {
    use super::{
        deck_contains_unique_cards, Action, BettingRound, Deck, LegalAction, NlheConfig, NlheGame,
        SeatPosition,
    };
    use rand::prelude::{Rng, SeedableRng, StdRng};

    fn choose_random_action(game: &NlheGame, legal: &[LegalAction], rng: &mut StdRng) -> Action {
        let picked = legal[rng.gen_range(0..legal.len())];
        match picked {
            LegalAction::Fold => Action::Fold,
            LegalAction::Check => Action::Check,
            LegalAction::Call { .. } => Action::Call,
            LegalAction::Bet { min, max } => Action::Bet(if min == max {
                min
            } else {
                rng.gen_range(min..=max)
            }),
            LegalAction::RaiseTo { min, max } => Action::RaiseTo(if min == max {
                min
            } else {
                rng.gen_range(min..=max)
            }),
            LegalAction::AllIn { total } => {
                let actor = game.current_actor().expect("actor expected");
                if total == game.players[actor].street_contribution {
                    Action::Check
                } else {
                    Action::AllIn
                }
            }
        }
    }

    #[test]
    fn ordered_deck_has_52_unique_cards() {
        let deck = Deck::new_ordered();
        assert_eq!(deck.remaining(), 52);
        assert!(deck_contains_unique_cards(&deck));
    }

    #[test]
    fn shuffled_decks_are_reproducible_for_same_seed() {
        let mut first = Deck::shuffled(42);
        let mut second = Deck::shuffled(42);

        for _ in 0..52 {
            assert_eq!(first.draw(), second.draw());
        }
    }

    #[test]
    fn six_max_position_order_is_stable() {
        assert_eq!(SeatPosition::SIX_MAX_ORDER.len(), 6);
        assert_eq!(SeatPosition::SIX_MAX_ORDER[0], SeatPosition::SmallBlind);
        assert_eq!(SeatPosition::SIX_MAX_ORDER[5], SeatPosition::Button);
    }

    #[test]
    fn preflop_starts_from_utg_in_six_max() {
        let game = NlheGame::new(NlheConfig::default(), 9).expect("game creation must work");
        assert_eq!(game.round, BettingRound::Preflop);
        assert_eq!(game.current_actor(), Some(2));

        let legal = game
            .legal_actions_for_current_actor()
            .expect("legal actions");
        assert!(legal.iter().any(|a| matches!(a, LegalAction::Fold)));
        assert!(legal.iter().any(|a| matches!(a, LegalAction::Call { .. })));
        assert!(legal
            .iter()
            .any(|a| matches!(a, LegalAction::RaiseTo { .. })));
    }

    #[test]
    fn random_playouts_preserve_total_chips() {
        let mut rng = StdRng::seed_from_u64(777);
        let config = NlheConfig {
            num_players: 6,
            starting_stack: 4_000,
            small_blind: 50,
            big_blind: 100,
        };
        let expected_total = config.starting_stack * config.num_players as u32;

        for hand_idx in 0..10_000 {
            let seed = rng.gen::<u64>();
            let mut game = NlheGame::new(config, seed).expect("game setup should not fail");

            let mut step_count = 0usize;
            while !game.is_complete() {
                step_count += 1;
                assert!(
                    step_count < 500,
                    "hand {hand_idx} exceeded action safety limit"
                );
                let actor = game.current_actor().expect("actor should exist");
                let legal = game.legal_actions(actor).expect("legal actions");
                assert!(!legal.is_empty(), "actor {actor} has no legal action");
                let action = choose_random_action(&game, &legal, &mut rng);
                game.apply_action(actor, action)
                    .expect("sampled legal action should apply");
            }

            assert_eq!(
                game.total_chips(),
                expected_total,
                "chip leak in hand {hand_idx}"
            );
            let summary = game.summary().expect("completed hand should have summary");
            assert_eq!(
                summary.payouts.iter().sum::<u32>(),
                summary.total_pot,
                "payout mismatch in hand {hand_idx}"
            );
        }
    }

    #[test]
    fn hu_postflop_first_actor_is_big_blind() {
        let mut game = NlheGame::new(
            NlheConfig {
                num_players: 2,
                starting_stack: 2_000,
                small_blind: 10,
                big_blind: 20,
            },
            1234,
        )
        .expect("game setup should work");
        assert_eq!(
            game.current_actor(),
            Some(0),
            "SB/button acts first preflop"
        );
        game.apply_action(0, Action::Call)
            .expect("SB should be able to complete to BB");
        assert_eq!(game.round, BettingRound::Preflop);
        assert_eq!(game.current_actor(), Some(1), "BB closes preflop action");
        game.apply_action(1, Action::Check)
            .expect("BB should be able to check after SB completes");
        assert_eq!(game.round, BettingRound::Flop);
        assert_eq!(
            game.current_actor(),
            Some(1),
            "BB should act first postflop in HU"
        );
    }

    #[test]
    fn legal_actions_do_not_duplicate_call_off_and_all_in() {
        let game = NlheGame::new(
            NlheConfig {
                num_players: 2,
                starting_stack: 20,
                small_blind: 10,
                big_blind: 20,
            },
            7,
        )
        .expect("game setup should work");
        let legal = game
            .legal_actions_for_current_actor()
            .expect("legal actions for SB preflop");
        assert!(
            legal
                .iter()
                .any(|a| matches!(a, LegalAction::Call { amount: 10 })),
            "SB should be able to call all-in for 10 chips"
        );
        assert!(
            !legal.iter().any(|a| matches!(a, LegalAction::AllIn { .. })),
            "all-in duplicate should be suppressed when call already commits full stack"
        );
    }

    #[test]
    fn legal_actions_do_not_duplicate_raise_to_max_and_all_in() {
        let game = NlheGame::new(
            NlheConfig {
                num_players: 2,
                starting_stack: 2_000,
                small_blind: 10,
                big_blind: 20,
            },
            77,
        )
        .expect("game setup should work");
        let legal = game
            .legal_actions_for_current_actor()
            .expect("legal actions for SB preflop");
        assert!(
            legal
                .iter()
                .any(|a| matches!(a, LegalAction::RaiseTo { .. })),
            "raise action should be available"
        );
        assert!(
            !legal.iter().any(|a| matches!(a, LegalAction::AllIn { .. })),
            "all-in duplicate should be suppressed when raise-to max already covers it"
        );
    }
}
