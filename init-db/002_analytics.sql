-- Analytics tables for FPL Fantasy God

-- Team results for form calculation
CREATE TABLE IF NOT EXISTS team_results (
    id SERIAL PRIMARY KEY,
    team_id INTEGER REFERENCES teams(id),
    fixture_id INTEGER REFERENCES fixtures(id),
    event INTEGER,
    opponent_id INTEGER REFERENCES teams(id),
    was_home BOOLEAN,
    goals_for INTEGER,
    goals_against INTEGER,
    result VARCHAR(1),  -- W/D/L
    points INTEGER,     -- 3/1/0
    clean_sheet BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(team_id, fixture_id)
);

-- Expected points cache for performance
CREATE TABLE IF NOT EXISTS expected_points_cache (
    player_id INTEGER PRIMARY KEY REFERENCES players(id),
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    base_xp DECIMAL(6,2),
    fixture_adjusted_xp DECIMAL(6,2),
    final_xp DECIMAL(6,2),
    xp_per_million DECIMAL(6,3),
    gw1_xp DECIMAL(6,2),
    gw2_xp DECIMAL(6,2),
    gw3_xp DECIMAL(6,2),
    gw4_xp DECIMAL(6,2),
    gw5_xp DECIMAL(6,2)
);

-- Price tracking for predictions
CREATE TABLE IF NOT EXISTS price_tracking (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(id),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    price INTEGER,
    net_transfers INTEGER,
    ownership_percent DECIMAL(5,2),
    predicted_change INTEGER  -- -1, 0, +1
);

CREATE INDEX IF NOT EXISTS idx_price_tracking_player ON price_tracking(player_id);
CREATE INDEX IF NOT EXISTS idx_price_tracking_recorded ON price_tracking(recorded_at);

-- Chip usage tracking (for analyzing chip timing)
CREATE TABLE IF NOT EXISTS chip_opportunities (
    id SERIAL PRIMARY KEY,
    gameweek INTEGER,
    chip_type VARCHAR(20),  -- TC, BB, FH, WC
    opportunity_score DECIMAL(6,2),
    dgw_teams INTEGER,
    bgw_teams INTEGER,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Player bogey teams (historical performance vs opponents)
CREATE TABLE IF NOT EXISTS player_opponent_history (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(id),
    opponent_id INTEGER REFERENCES teams(id),
    games_played INTEGER DEFAULT 0,
    total_points INTEGER DEFAULT 0,
    goals INTEGER DEFAULT 0,
    assists INTEGER DEFAULT 0,
    avg_points DECIMAL(5,2),
    is_bogey_team BOOLEAN DEFAULT FALSE,  -- True if performs poorly
    is_favourite BOOLEAN DEFAULT FALSE,   -- True if performs well
    UNIQUE(player_id, opponent_id)
);

CREATE INDEX IF NOT EXISTS idx_player_opponent ON player_opponent_history(player_id);

-- Double/Blank gameweek tracking
CREATE TABLE IF NOT EXISTS gameweek_special (
    id SERIAL PRIMARY KEY,
    gameweek INTEGER UNIQUE,
    is_double BOOLEAN DEFAULT FALSE,
    is_blank BOOLEAN DEFAULT FALSE,
    dgw_team_count INTEGER DEFAULT 0,
    bgw_team_count INTEGER DEFAULT 0,
    notes TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Views for common analytics queries

-- View: Player form with trend
CREATE OR REPLACE VIEW v_player_form_trend AS
SELECT
    p.id,
    p.web_name,
    p.team_id,
    t.short_name as team,
    p.element_type,
    p.form as current_form,
    p.now_cost,
    p.selected_by_percent,
    ps_recent.form as form_1gw_ago,
    ps_old.form as form_3gw_ago,
    COALESCE(p.form - ps_old.form, 0) as form_change
FROM players p
JOIN teams t ON p.team_id = t.id
LEFT JOIN LATERAL (
    SELECT form FROM player_snapshots
    WHERE player_id = p.id
    ORDER BY gameweek DESC
    LIMIT 1 OFFSET 1
) ps_recent ON true
LEFT JOIN LATERAL (
    SELECT form FROM player_snapshots
    WHERE player_id = p.id
    ORDER BY gameweek DESC
    LIMIT 1 OFFSET 3
) ps_old ON true;

-- View: Team fixture difficulty
CREATE OR REPLACE VIEW v_team_fixture_difficulty AS
SELECT
    t.id as team_id,
    t.name as team_name,
    t.short_name,
    COUNT(f.id) as upcoming_fixtures,
    AVG(CASE WHEN f.team_h = t.id THEN f.team_h_difficulty ELSE f.team_a_difficulty END) as avg_difficulty
FROM teams t
LEFT JOIN fixtures f ON (f.team_h = t.id OR f.team_a = t.id) AND f.finished = false
GROUP BY t.id, t.name, t.short_name;

-- View: Price change candidates
CREATE OR REPLACE VIEW v_price_change_candidates AS
SELECT
    p.id,
    p.web_name,
    t.short_name as team,
    p.now_cost,
    p.transfers_in_event,
    p.transfers_out_event,
    (p.transfers_in_event - p.transfers_out_event) as net_transfers,
    p.selected_by_percent,
    p.status,
    CASE
        WHEN (p.transfers_in_event - p.transfers_out_event) > 100000 THEN 'likely_rise'
        WHEN (p.transfers_in_event - p.transfers_out_event) < -100000 THEN 'likely_fall'
        ELSE 'stable'
    END as price_prediction
FROM players p
JOIN teams t ON p.team_id = t.id
WHERE ABS(p.transfers_in_event - p.transfers_out_event) > 50000
ORDER BY net_transfers DESC;

-- View: xG overperformers (regression candidates)
CREATE OR REPLACE VIEW v_xg_overperformers AS
SELECT
    p.id,
    p.web_name,
    t.short_name as team,
    p.element_type,
    p.goals_scored,
    p.expected_goals,
    (p.goals_scored - COALESCE(p.expected_goals, 0)) as goals_vs_xg,
    p.assists,
    p.expected_assists,
    (p.assists - COALESCE(p.expected_assists, 0)) as assists_vs_xa,
    (p.goals_scored - COALESCE(p.expected_goals, 0) + p.assists - COALESCE(p.expected_assists, 0)) as total_overperformance
FROM players p
JOIN teams t ON p.team_id = t.id
WHERE p.minutes > 450
ORDER BY total_overperformance DESC;

-- Function to populate team results from fixtures
CREATE OR REPLACE FUNCTION populate_team_results()
RETURNS void AS $$
BEGIN
    INSERT INTO team_results (team_id, fixture_id, event, opponent_id, was_home, goals_for, goals_against, result, points, clean_sheet)
    SELECT
        t.id as team_id,
        f.id as fixture_id,
        f.event,
        CASE WHEN f.team_h = t.id THEN f.team_a ELSE f.team_h END as opponent_id,
        (f.team_h = t.id) as was_home,
        CASE WHEN f.team_h = t.id THEN f.team_h_score ELSE f.team_a_score END as goals_for,
        CASE WHEN f.team_h = t.id THEN f.team_a_score ELSE f.team_h_score END as goals_against,
        CASE
            WHEN (CASE WHEN f.team_h = t.id THEN f.team_h_score ELSE f.team_a_score END) >
                 (CASE WHEN f.team_h = t.id THEN f.team_a_score ELSE f.team_h_score END) THEN 'W'
            WHEN (CASE WHEN f.team_h = t.id THEN f.team_h_score ELSE f.team_a_score END) <
                 (CASE WHEN f.team_h = t.id THEN f.team_a_score ELSE f.team_h_score END) THEN 'L'
            ELSE 'D'
        END as result,
        CASE
            WHEN (CASE WHEN f.team_h = t.id THEN f.team_h_score ELSE f.team_a_score END) >
                 (CASE WHEN f.team_h = t.id THEN f.team_a_score ELSE f.team_h_score END) THEN 3
            WHEN (CASE WHEN f.team_h = t.id THEN f.team_h_score ELSE f.team_a_score END) <
                 (CASE WHEN f.team_h = t.id THEN f.team_a_score ELSE f.team_h_score END) THEN 0
            ELSE 1
        END as points,
        (CASE WHEN f.team_h = t.id THEN f.team_a_score ELSE f.team_h_score END) = 0 as clean_sheet
    FROM fixtures f
    CROSS JOIN teams t
    WHERE f.finished = true
      AND (f.team_h = t.id OR f.team_a = t.id)
      AND f.team_h_score IS NOT NULL
    ON CONFLICT (team_id, fixture_id) DO UPDATE SET
        goals_for = EXCLUDED.goals_for,
        goals_against = EXCLUDED.goals_against,
        result = EXCLUDED.result,
        points = EXCLUDED.points,
        clean_sheet = EXCLUDED.clean_sheet;
END;
$$ LANGUAGE plpgsql;

-- Function to update player opponent history
CREATE OR REPLACE FUNCTION update_player_opponent_history()
RETURNS void AS $$
BEGIN
    INSERT INTO player_opponent_history (player_id, opponent_id, games_played, total_points, goals, assists, avg_points)
    SELECT
        ph.player_id,
        ph.opponent_team,
        COUNT(*) as games_played,
        SUM(COALESCE(ph.total_points, 0)) as total_points,
        SUM(COALESCE(ph.goals_scored, 0)) as goals,
        SUM(COALESCE(ph.assists, 0)) as assists,
        AVG(COALESCE(ph.total_points, 0)) as avg_points
    FROM player_history ph
    WHERE ph.opponent_team IS NOT NULL
    GROUP BY ph.player_id, ph.opponent_team
    ON CONFLICT (player_id, opponent_id) DO UPDATE SET
        games_played = EXCLUDED.games_played,
        total_points = EXCLUDED.total_points,
        goals = EXCLUDED.goals,
        assists = EXCLUDED.assists,
        avg_points = EXCLUDED.avg_points;

    -- Mark bogey teams (significantly below average)
    UPDATE player_opponent_history poh
    SET is_bogey_team = (poh.avg_points < pavg.player_avg * 0.7)
    FROM (
        SELECT player_id, AVG(avg_points) as player_avg
        FROM player_opponent_history
        GROUP BY player_id
    ) pavg
    WHERE poh.player_id = pavg.player_id AND poh.games_played >= 2;

    -- Mark favourite teams (significantly above average)
    UPDATE player_opponent_history poh
    SET is_favourite = (poh.avg_points > pavg.player_avg * 1.3)
    FROM (
        SELECT player_id, AVG(avg_points) as player_avg
        FROM player_opponent_history
        GROUP BY player_id
    ) pavg
    WHERE poh.player_id = pavg.player_id AND poh.games_played >= 2;
END;
$$ LANGUAGE plpgsql;
