-- Initialize AURA Persistence Database
-- Enables pgvector and creates optimized schemas

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "btree_gist";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create custom types
CREATE TYPE state_type AS ENUM (
    'component_state',
    'liquid_network', 
    'memory_tier',
    'metabolic_budget',
    'system_config',
    'agent_memory',
    'neural_checkpoint',
    'tda_cache',
    'swarm_state'
);

-- Create main states table with causality support
CREATE TABLE IF NOT EXISTS states (
    state_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    state_type state_type NOT NULL,
    component_id TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    branch_id TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Data storage
    data JSONB,
    compressed_data BYTEA,
    embedding vector(768),  -- For semantic search
    
    -- Causality tracking
    causes TEXT[],
    effects TEXT[], 
    counterfactuals JSONB,
    confidence FLOAT DEFAULT 1.0,
    energy_cost FLOAT DEFAULT 0.0,
    decision_path JSONB,
    
    -- Metadata
    checksum TEXT NOT NULL,
    gpu_cached BOOLEAN DEFAULT FALSE,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_states_component ON states(component_id, version DESC);
CREATE INDEX idx_states_type_time ON states(state_type, timestamp DESC);
CREATE INDEX idx_states_branch ON states(branch_id) WHERE branch_id IS NOT NULL;
CREATE INDEX idx_states_causes ON states USING GIN(causes);
CREATE INDEX idx_states_effects ON states USING GIN(effects);
CREATE INDEX idx_states_embedding ON states USING ivfflat(embedding vector_cosine_ops);
CREATE INDEX idx_states_data ON states USING GIN(data jsonb_path_ops);
CREATE INDEX idx_states_access ON states(last_accessed DESC);

-- Causal edges for graph traversal
CREATE TABLE IF NOT EXISTS causal_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    from_state UUID REFERENCES states(state_id) ON DELETE CASCADE,
    to_state UUID REFERENCES states(state_id) ON DELETE CASCADE,
    edge_type TEXT NOT NULL,
    weight FLOAT DEFAULT 1.0,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_edges_from ON causal_edges(from_state);
CREATE INDEX idx_edges_to ON causal_edges(to_state);
CREATE INDEX idx_edges_type ON causal_edges(edge_type);

-- Version history for branches
CREATE TABLE IF NOT EXISTS version_history (
    component_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    branch_id TEXT,
    merged_from TEXT,
    merge_confidence FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (component_id, version)
);

CREATE INDEX idx_version_branch ON version_history(branch_id);
CREATE INDEX idx_version_time ON version_history(created_at DESC);

-- Speculative branches tracking
CREATE TABLE IF NOT EXISTS branches (
    branch_id TEXT PRIMARY KEY,
    component_id TEXT NOT NULL,
    parent_branch TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    merged_at TIMESTAMPTZ,
    status TEXT DEFAULT 'active',
    performance_metrics JSONB,
    exploration_strategy TEXT
);

CREATE INDEX idx_branches_component ON branches(component_id);
CREATE INDEX idx_branches_status ON branches(status);

-- Agent memory with conversation history
CREATE TABLE IF NOT EXISTS agent_memories (
    memory_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id TEXT NOT NULL,
    conversation_id TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    
    -- Memory content
    message JSONB NOT NULL,
    embedding vector(768),
    
    -- Causal context
    decision_context JSONB,
    tool_calls JSONB,
    outcomes JSONB,
    
    -- Metadata
    importance_score FLOAT DEFAULT 0.5,
    access_frequency INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_agent_memories_agent ON agent_memories(agent_id, timestamp DESC);
CREATE INDEX idx_agent_memories_conversation ON agent_memories(conversation_id);
CREATE INDEX idx_agent_memories_embedding ON agent_memories USING ivfflat(embedding vector_cosine_ops);
CREATE INDEX idx_agent_memories_importance ON agent_memories(importance_score DESC);

-- Neural network checkpoints
CREATE TABLE IF NOT EXISTS neural_checkpoints (
    checkpoint_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name TEXT NOT NULL,
    version INTEGER NOT NULL,
    epoch INTEGER,
    
    -- Model data
    architecture JSONB NOT NULL,
    weights_location TEXT NOT NULL,  -- S3/MinIO path
    optimizer_state JSONB,
    
    -- Training context
    training_metrics JSONB,
    validation_metrics JSONB,
    hyperparameters JSONB,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    training_time_seconds FLOAT,
    gpu_hours FLOAT
);

CREATE INDEX idx_checkpoints_model ON neural_checkpoints(model_name, version DESC);
CREATE INDEX idx_checkpoints_metrics ON neural_checkpoints USING GIN(validation_metrics jsonb_path_ops);

-- TDA computation cache
CREATE TABLE IF NOT EXISTS tda_cache (
    cache_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    data_hash TEXT NOT NULL,
    computation_type TEXT NOT NULL,
    
    -- Results
    persistence_diagram JSONB,
    betti_numbers JSONB,
    persistence_image BYTEA,
    
    -- Computation details
    parameters JSONB,
    computation_time_ms FLOAT,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_tda_cache_hash ON tda_cache(data_hash, computation_type);
CREATE INDEX idx_tda_cache_access ON tda_cache(last_accessed DESC);

-- Swarm collective state
CREATE TABLE IF NOT EXISTS swarm_states (
    state_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    swarm_id TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    
    -- Swarm data
    agent_positions JSONB,
    collective_knowledge JSONB,
    consensus_state JSONB,
    pheromone_trails JSONB,
    
    -- Performance
    collective_fitness FLOAT,
    diversity_score FLOAT,
    convergence_rate FLOAT,
    
    -- Metadata
    agent_count INTEGER,
    iteration INTEGER
);

CREATE INDEX idx_swarm_states_swarm ON swarm_states(swarm_id, timestamp DESC);
CREATE INDEX idx_swarm_states_fitness ON swarm_states(collective_fitness DESC);

-- Create functions for causal analysis
CREATE OR REPLACE FUNCTION get_causal_depth(start_state UUID) 
RETURNS INTEGER AS $$
DECLARE
    depth INTEGER;
BEGIN
    WITH RECURSIVE causal_chain AS (
        SELECT state_id, causes, 0 as depth
        FROM states
        WHERE state_id = start_state
        
        UNION ALL
        
        SELECT s.state_id, s.causes, cc.depth + 1
        FROM states s
        JOIN causal_chain cc ON s.state_id = ANY(cc.causes::uuid[])
        WHERE cc.depth < 100  -- Prevent infinite recursion
    )
    SELECT MAX(depth) INTO depth FROM causal_chain;
    
    RETURN COALESCE(depth, 0);
END;
$$ LANGUAGE plpgsql;

-- Function to calculate information gain
CREATE OR REPLACE FUNCTION calculate_information_gain(
    before_state JSONB,
    after_state JSONB
) RETURNS FLOAT AS $$
DECLARE
    gain FLOAT;
BEGIN
    -- Simple entropy-based calculation
    -- In practice, this would be more sophisticated
    gain := jsonb_array_length(after_state) - jsonb_array_length(before_state);
    RETURN ABS(gain);
END;
$$ LANGUAGE plpgsql;

-- Trigger to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_states_updated_at
    BEFORE UPDATE ON states
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Create views for easy querying
CREATE OR REPLACE VIEW latest_states AS
SELECT DISTINCT ON (component_id, state_type)
    state_id,
    state_type,
    component_id,
    version,
    branch_id,
    timestamp,
    gpu_cached,
    confidence
FROM states
ORDER BY component_id, state_type, version DESC;

CREATE OR REPLACE VIEW causal_graph_view AS
SELECT 
    e.id as edge_id,
    e.edge_type,
    e.weight,
    s1.component_id as from_component,
    s1.state_type as from_type,
    s2.component_id as to_component,
    s2.state_type as to_type,
    e.created_at
FROM causal_edges e
JOIN states s1 ON e.from_state = s1.state_id
JOIN states s2 ON e.to_state = s2.state_id;

-- Performance optimization settings
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';
ALTER SYSTEM SET maintenance_work_mem = '512MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;
ALTER SYSTEM SET work_mem = '256MB';
ALTER SYSTEM SET min_wal_size = '1GB';
ALTER SYSTEM SET max_wal_size = '4GB';

-- Create initial user and permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO aura;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO aura;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO aura;