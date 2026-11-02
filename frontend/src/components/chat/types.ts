export type Citation = {
  id?: string;
  label: string;
  ordinal?: number;
  source_type: string;
  title: string;
  filename?: string;
  document_id: string | null;
  kb_id?: string | null;
  chunk_id?: string | null;
  locator: Record<string, any>;
  locator_text?: string;
  snippet: string;
  score?: number | null;
  scores?: Record<string, number>;
  flagged?: boolean;
  cited: boolean;
  accessible?: boolean;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: string;
  provider?: string | null;
  model?: string | null;
  confidence?: number | null;
  grounded?: boolean | null;
  retrieval_ms?: number | null;
  generation_ms?: number | null;
  tokens_in?: number;
  tokens_out?: number;
  metadata?: {
    mode?: string;
    follow_ups?: string[];
    unsupported_sentences?: string[];
    invalid_citations?: string[];
    trace?: Record<string, any>;
    support_ratio?: number | null;
    error?: string | null;
  };
  citations: Citation[];
  created_at?: string;
  /* live-only */
  streaming?: boolean;
  stage?: string;
  liveSources?: Citation[];
};

export type Conversation = {
  id: string;
  title: string;
  kb_ids: string[];
  agent_id: string | null;
  own: boolean;
  messages: ChatMessage[];
};

export type KB = { id: string; name: string; description: string; access: string; stats: { documents: number; chunks: number } };

export type RetrievalSettings = {
  mode: "hybrid" | "dense" | "sparse" | "keyword";
  top_k: number;
  rerank: "lexical" | "none";
  mmr: boolean;
  query_expansion: "prf" | "none";
  compression: boolean;
};

export type ChatFilters = {
  file_types: string[];
  tags: string[];
  source_types: string[];
  created_from: string;
  created_to: string;
};
