import type { GraphEdge, GraphNode } from "@/components/connections/types";

/**
 * Deterministic layered layout that follows the data path:
 * organization → workspaces & sources → ingestion → knowledge bases →
 * embeddings → vector index → RAG engine → models → agents.
 * Stable positions mean nodes don't jump around when live updates arrive.
 *
 * `LR` runs the path left → right (wide canvases); `TB` runs it top → bottom
 * (narrow or portrait canvases), so the graph stays legible on a phone.
 * Empty layers are skipped so the graph never has holes.
 */
const COLUMN: Record<string, number> = {
  organization: 0,
  workspace: 1,
  database: 1,
  warehouse: 1,
  vector_db: 1,
  integration: 1,
  document_source: 1,
  group: 1,
  pipeline: 2,
  knowledge_base: 3,
  embedding: 4,
  vector_store: 5,
  rag: 6,
  ai_model: 7,
  agent: 8,
};

export type Orientation = "LR" | "TB";

/** Node geometry (flow units). Kept in sync with `.cx-node` in connections.css. */
export const NODE_W = 148;
export const TILE = 46;
export const TILE_LG = 58;
/** Tile + label block, used to estimate the graph extent before measuring. */
export const NODE_H = 94;

const LR_COL = 158; // distance between layers
const LR_ROW = 108; // distance between nodes inside a layer
const TB_COL = 158;
const TB_ROW = 142;

const ORDER = ["workspace", "document_source", "database", "warehouse", "vector_db", "integration", "group"];

function layers(nodes: GraphNode[]): GraphNode[][] {
  const cols = new Map<number, GraphNode[]>();
  for (const n of nodes) {
    const c = COLUMN[n.type] ?? 4;
    if (!cols.has(c)) cols.set(c, []);
    cols.get(c)!.push(n);
  }
  // Order knowledge bases by the workspace they belong to, sources by kind then name.
  const wsIndex = new Map<string, number>();
  (cols.get(1) ?? []).filter((n) => n.type === "workspace").forEach((n, i) => wsIndex.set(n.id.split(":")[1], i));
  for (const [c, list] of cols) {
    list.sort((a, b) => {
      if (c === 1) return ORDER.indexOf(a.type) - ORDER.indexOf(b.type) || a.label.localeCompare(b.label);
      if (c === 3) return (wsIndex.get(a.data.workspace_id) ?? 99) - (wsIndex.get(b.data.workspace_id) ?? 99) || a.label.localeCompare(b.label);
      return a.label.localeCompare(b.label);
    });
  }
  return [...cols.keys()].sort((a, b) => a - b).map((k) => cols.get(k)!);
}

export function layout(nodes: GraphNode[], edges: GraphEdge[], orientation: Orientation = "LR"): Record<string, { x: number; y: number }> {
  const ls = layers(nodes);
  const tallest = Math.max(1, ...ls.map((l) => l.length));
  const pos: Record<string, { x: number; y: number }> = {};
  ls.forEach((list, li) => {
    list.forEach((n, i) => {
      const slot = (tallest - list.length) / 2 + i;
      const tile = n.type === "organization" ? TILE_LG : TILE;
      // (cx, cy) is the centre of the node's tile; positions are the node's top-left.
      const cx = orientation === "LR" ? li * LR_COL : slot * TB_COL;
      const cy = orientation === "LR" ? slot * LR_ROW : li * TB_ROW;
      pos[n.id] = { x: cx - NODE_W / 2, y: cy - tile / 2 };
    });
  });
  void edges;
  return pos;
}

/** Approximate graph size for an orientation, before React Flow measures it. */
export function extent(nodes: GraphNode[], orientation: Orientation): { w: number; h: number } {
  const ls = layers(nodes);
  const depth = Math.max(1, ls.length);
  const breadth = Math.max(1, ...ls.map((l) => l.length));
  return orientation === "LR"
    ? { w: (depth - 1) * LR_COL + NODE_W, h: (breadth - 1) * LR_ROW + NODE_H }
    : { w: (breadth - 1) * TB_COL + NODE_W, h: (depth - 1) * TB_ROW + NODE_H };
}
