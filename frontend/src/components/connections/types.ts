export type NodeStatus = "healthy" | "syncing" | "warning" | "failed" | "paused" | "disabled" | "idle";

export type GraphNode = {
  id: string;
  type: string;
  label: string;
  status: NodeStatus;
  data: Record<string, any>;
};

export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  kind: string;
  status: "healthy" | "active" | "failed" | "paused" | "disabled" | "idle";
  data: Record<string, any>;
};

export type GraphSummary = {
  total_sources: number;
  active: number;
  disconnected: number;
  failed: number;
  syncing: number;
  documents: number;
  records: number;
  chunks: number;
  active_jobs: number;
  storage_bytes: number;
  last_successful_sync: string | null;
};

export type GraphResponse = { nodes: GraphNode[]; edges: GraphEdge[]; grouped: boolean; summary: GraphSummary; generated_at: string };

export const SOURCE_TYPES = new Set(["database", "warehouse", "vector_db", "integration", "document_source", "group"]);

export const CATEGORY_FILTERS: { key: string; label: string; match: (n: GraphNode) => boolean }[] = [
  { key: "all", label: "All", match: () => true },
  { key: "databases", label: "Databases", match: (n) => n.type === "database" },
  { key: "vector", label: "Vector DBs", match: (n) => n.type === "vector_db" || n.type === "vector_store" },
  { key: "warehouses", label: "Warehouses", match: (n) => n.type === "warehouse" },
  { key: "documents", label: "Document sources", match: (n) => n.type === "document_source" },
  { key: "integrations", label: "Integrations", match: (n) => n.type === "integration" },
  { key: "kbs", label: "Knowledge bases", match: (n) => n.type === "knowledge_base" },
  { key: "ai", label: "AI services", match: (n) => ["embedding", "rag", "ai_model", "agent", "pipeline"].includes(n.type) },
];

export const STATUS_FILTERS: { key: string; label: string; match: (n: GraphNode) => boolean }[] = [
  { key: "active", label: "Active", match: (n) => n.status === "healthy" || n.status === "syncing" },
  { key: "failed", label: "Failed", match: (n) => n.status === "failed" || n.status === "warning" },
  { key: "syncing", label: "Synchronizing", match: (n) => n.status === "syncing" },
];
