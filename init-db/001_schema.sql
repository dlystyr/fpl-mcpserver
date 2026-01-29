-- FPL Database Schema

-- Teams
CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    short_name VARCHAR(10) NOT NULL,
    code INTEGER,
    strength INTEGER,
    strength_overall_home INTEGER,
    strength_overall_away INTEGER,
    strength_attack_home INTEGER,
    strength_attack_away INTEGER,
    strength_defence_home INTEGER,
    strength_defence_away INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Players
CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY,
    code INTEGER,
    first_name VARCHAR(100),
    second_name VARCHAR(100),
    web_name VARCHAR(100) NOT NULL,
    team_id INTEGER REFERENCES teams(id),
    element_type INTEGER NOT NULL, -- 1=GK, 2=DEF, 3=MID, 4=FWD
    now_cost INTEGER,
    selected_by_percent DECIMAL(5,2),
    transfers_in_event INTEGER DEFAULT 0,
    transfers_out_event INTEGER DEFAULT 0,
    form DECIMAL(4,2),
    points_per_game DECIMAL(4,2),
    total_points INTEGER DEFAULT 0,
    minutes INTEGER DEFAULT 0,
    goals_scored INTEGER DEFAULT 0,
    assists INTEGER DEFAULT 0,
    clean_sheets INTEGER DEFAULT 0,
    goals_conceded INTEGER DEFAULT 0,
    own_goals INTEGER DEFAULT 0,
    penalties_saved INTEGER DEFAULT 0,
    penalties_missed INTEGER DEFAULT 0,
    yellow_cards INTEGER DEFAULT 0,
    red_cards INTEGER DEFAULT 0,
    saves INTEGER DEFAULT 0,
    bonus INTEGER DEFAULT 0,
    bps INTEGER DEFAULT 0,
    expected_goals DECIMAL(6,2),
    expected_assists DECIMAL(6,2),
    expected_goal_involvements DECIMAL(6,2),
    expected_goals_conceded DECIMAL(6,2),
    influence DECIMAL(6,2),
    creativity DECIMAL(6,2),
    threat DECIMAL(6,2),
    ict_index DECIMAL(6,2),
    starts INTEGER DEFAULT 0,
    status VARCHAR(10) DEFAULT 'a',
    chance_of_playing_next_round INTEGER,
    chance_of_playing_this_round INTEGER,
    news TEXT,
    news_added TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_id);
CREATE INDEX IF NOT EXISTS idx_players_element_type ON players(element_type);
CREATE INDEX IF NOT EXISTS idx_players_form ON players(form DESC);
CREATE INDEX IF NOT EXISTS idx_players_total_points ON players(total_points DESC);

-- Gameweeks/Events
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    deadline_time TIMESTAMP,
    finished BOOLEAN DEFAULT FALSE,
    is_current BOOLEAN DEFAULT FALSE,
    is_next BOOLEAN DEFAULT FALSE,
    is_previous BOOLEAN DEFAULT FALSE,
    average_entry_score INTEGER,
    highest_score INTEGER,
    most_selected INTEGER,
    most_captained INTEGER,
    most_vice_captained INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Fixtures
CREATE TABLE IF NOT EXISTS fixtures (
    id INTEGER PRIMARY KEY,
    code INTEGER,
    event INTEGER REFERENCES events(id),
    team_h INTEGER REFERENCES teams(id),
    team_a INTEGER REFERENCES teams(id),
    team_h_score INTEGER,
    team_a_score INTEGER,
    finished BOOLEAN DEFAULT FALSE,
    kickoff_time TIMESTAMP,
    team_h_difficulty INTEGER,
    team_a_difficulty INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fixtures_event ON fixtures(event);
CREATE INDEX IF NOT EXISTS idx_fixtures_team_h ON fixtures(team_h);
CREATE INDEX IF NOT EXISTS idx_fixtures_team_a ON fixtures(team_a);

-- Player History (per gameweek stats)
CREATE TABLE IF NOT EXISTS player_history (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(id),
    fixture_id INTEGER,
    event INTEGER REFERENCES events(id),
    opponent_team INTEGER REFERENCES teams(id),
    was_home BOOLEAN,
    total_points INTEGER DEFAULT 0,
    minutes INTEGER DEFAULT 0,
    goals_scored INTEGER DEFAULT 0,
    assists INTEGER DEFAULT 0,
    clean_sheets INTEGER DEFAULT 0,
    goals_conceded INTEGER DEFAULT 0,
    own_goals INTEGER DEFAULT 0,
    penalties_saved INTEGER DEFAULT 0,
    penalties_missed INTEGER DEFAULT 0,
    yellow_cards INTEGER DEFAULT 0,
    red_cards INTEGER DEFAULT 0,
    saves INTEGER DEFAULT 0,
    bonus INTEGER DEFAULT 0,
    bps INTEGER DEFAULT 0,
    expected_goals DECIMAL(6,2),
    expected_assists DECIMAL(6,2),
    expected_goal_involvements DECIMAL(6,2),
    expected_goals_conceded DECIMAL(6,2),
    influence DECIMAL(6,2),
    creativity DECIMAL(6,2),
    threat DECIMAL(6,2),
    ict_index DECIMAL(6,2),
    value INTEGER,
    selected INTEGER,
    UNIQUE(player_id, event)
);

CREATE INDEX IF NOT EXISTS idx_player_history_player ON player_history(player_id);
CREATE INDEX IF NOT EXISTS idx_player_history_event ON player_history(event);

-- Player Snapshots (per gameweek for trend tracking)
CREATE TABLE IF NOT EXISTS player_snapshots (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(id),
    gameweek INTEGER NOT NULL,
    now_cost INTEGER,
    selected_by_percent DECIMAL(5,2),
    form DECIMAL(4,2),
    points_per_game DECIMAL(4,2),
    total_points INTEGER,
    minutes INTEGER,
    goals_scored INTEGER,
    assists INTEGER,
    clean_sheets INTEGER,
    goals_conceded INTEGER,
    bonus INTEGER,
    bps INTEGER,
    expected_goals DECIMAL(6,2),
    expected_assists DECIMAL(6,2),
    expected_goal_involvements DECIMAL(6,2),
    influence DECIMAL(6,2),
    creativity DECIMAL(6,2),
    threat DECIMAL(6,2),
    ict_index DECIMAL(6,2),
    transfers_in_event INTEGER,
    transfers_out_event INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(player_id, gameweek)
);

CREATE INDEX IF NOT EXISTS idx_player_snapshots_player ON player_snapshots(player_id);
CREATE INDEX IF NOT EXISTS idx_player_snapshots_gw ON player_snapshots(gameweek);

-- Sync Log
CREATE TABLE IF NOT EXISTS sync_log (
    id SERIAL PRIMARY KEY,
    sync_type VARCHAR(50) NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR(20),
    records_updated INTEGER DEFAULT 0,
    error_message TEXT
);
