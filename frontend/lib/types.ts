export interface ModelConfig {
  vocab_size: number;
  hidden_dim: number;
  n_heads: number;
  context_layers: number;
  context_ff_mult: number;
  max_seq_len: number;
  max_steps: number;
  min_steps: number;
  halt_epsilon: number;
  dropout: number;
  ff_mult: number;
  memory_dim: number;
  memory_kind: string;
  memory_inner_steps: number;
  memory_inner_lr: number;
  memory_inner_optimizer?: string;
  memory_update_every?: number;
  memory_decay?: number;
  memory_clip?: number;
  memory_weight_decay?: number;
  memory_hidden_mult?: number;
  [key: string]: unknown;
}

export interface CheckpointMeta {
  checkpoint_id: string;
  created_at: number;
  task?: string;
  format?: string;
  path?: string;
  model_config?: Partial<ModelConfig>;
  training_config?: Record<string, unknown>;
  metrics?: Record<string, any>;
}

export interface Job {
  id: string;
  kind: "train" | "benchmark" | string;
  status: string;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  config: Record<string, any>;
  progress: number;
  message: string | null;
  result: any;
  error: string | null;
  cancel_requested: boolean;
}

export interface JobEvent {
  type: string;
  [key: string]: any;
}

export interface BenchmarkReportSummary {
  run_id: string;
  suite: string;
  created_at: number;
  duration_seconds: number;
  device: string;
  checkpoint_id: string | null;
  summary: Record<string, any>;
  path: string;
}

export interface MemorySessionSummary {
  session_id: string;
  name: string;
  created_at: number;
  documents: Array<Record<string, any>>;
  chunks: Array<Record<string, any>>;
  chunk_count?: number;
  record_count?: number;
  state_updates?: number;
  memory_dim?: number;
  memory_kind?: string;
  fast_parameters?: number;
  memory_state_bytes?: number;
  total_state_bytes?: number;
  capacity_chunks?: number;
  adapt_seconds?: number;
  state_bytes?: number;
  chunk_state_bytes?: number;
  metadata?: Record<string, any>;
  retrieval_counts?: Record<string, number>;
  last_query?: Record<string, any> | null;
  [key: string]: any;
}

export interface ModelSummary {
  checkpoint_id: string | null;
  default_checkpoint_id: string | null;
  device: string;
  model_config: ModelConfig;
  training_config: Record<string, any> | null;
  task: string | null;
  metrics: Record<string, any>;
  parameters_total: number;
  parameters_trainable: number;
  parameters_frozen: number;
  memory_initial_parameters: number;
  module_breakdown: Array<{ name: string; parameters: number }>;
}
