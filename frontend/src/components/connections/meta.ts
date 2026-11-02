import { Bot, Boxes, Brain, Building2, Cable, Cpu, Database, FileStack, FolderTree, Layers, Library, type LucideIcon, Server, Sparkles, Warehouse, Workflow } from "lucide-react";

import type { NodeStatus } from "@/components/connections/types";
import type { Tone } from "@/components/ui/primitives";

/** Presentation metadata shared by the map, the list and the detail panel. */

export const NODE_ICON: Record<string, LucideIcon> = {
  organization: Building2,
  workspace: FolderTree,
  database: Database,
  warehouse: Warehouse,
  vector_db: Boxes,
  integration: Cable,
  document_source: FileStack,
  group: Layers,
  pipeline: Workflow,
  knowledge_base: Library,
  embedding: Cpu,
  vector_store: Server,
  rag: Sparkles,
  ai_model: Brain,
  agent: Bot,
};

/**
 * Visual families along the data path: neutral sources → processing →
 * knowledge (soft iris) → intelligence (solid iris). Restrained, but each
 * stage of the path reads differently at a glance.
 */
export type Family = "org" | "source" | "process" | "knowledge" | "intelligence";

export const NODE_FAMILY: Record<string, Family> = {
  organization: "org",
  workspace: "source",
  database: "source",
  warehouse: "source",
  vector_db: "source",
  integration: "source",
  document_source: "source",
  group: "source",
  pipeline: "process",
  embedding: "process",
  vector_store: "process",
  knowledge_base: "knowledge",
  rag: "intelligence",
  ai_model: "intelligence",
  agent: "intelligence",
};

export const TYPE_LABEL: Record<string, string> = {
  organization: "Organization",
  workspace: "Workspace",
  database: "Database",
  warehouse: "Warehouse",
  vector_db: "Vector DB",
  integration: "Integration",
  document_source: "File uploads",
  group: "Source group",
  pipeline: "Ingestion pipeline",
  knowledge_base: "Knowledge base",
  embedding: "Embedding model",
  vector_store: "Vector index",
  rag: "RAG engine",
  ai_model: "AI model",
  agent: "Agent",
};

/** Same tones as `StatusBadge`, so the map, list and detail panel agree. */
export const STATUS_TONE: Record<NodeStatus, Tone> = {
  healthy: "ok",
  syncing: "info",
  warning: "warn",
  failed: "danger",
  paused: "warn",
  disabled: "neutral",
  idle: "neutral",
};

export const STATUS_LABEL: Record<NodeStatus, string> = {
  healthy: "Healthy",
  syncing: "Synchronizing",
  warning: "Warning",
  failed: "Failed",
  paused: "Paused",
  disabled: "Disabled",
  idle: "Idle",
};

const TONE_VAR: Record<Tone, string> = { ok: "--ok", info: "--info", warn: "--warn", danger: "--danger", neutral: "--subtle", accent: "--accent" };

/** `var(--ok)` style reference, for `hsl(${toneVar(t)} / a)` in inline styles. */
export function toneVar(t: Tone): string {
  return `var(${TONE_VAR[t]})`;
}

export function statusTone(s: string | null | undefined): Tone {
  return STATUS_TONE[s as NodeStatus] ?? "neutral";
}

export const EVENT_TONE: Record<string, Tone> = { error: "danger", warning: "warn", success: "ok", info: "info" };

/** Small deterministic hash, used to desynchronise ambient animations. */
export function hash(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return Math.abs(h);
}
