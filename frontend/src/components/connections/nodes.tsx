"use client";

import { Handle, Position, useStore, type NodeProps } from "@xyflow/react";
import { Boxes } from "lucide-react";
import { memo } from "react";

import type { Orientation } from "@/components/connections/layout";
import { hash, NODE_FAMILY, NODE_ICON, STATUS_TONE, toneVar } from "@/components/connections/meta";
import type { NodeStatus } from "@/components/connections/types";
import { cn, formatNumber } from "@/lib/utils";

export type MapNodeData = {
  label: string;
  kind: string;
  status: NodeStatus;
  meta: Record<string, any>;
  showLabels: boolean;
  animate: boolean;
  dimmed: boolean;
  flash?: "ok" | "error" | "new" | null;
  orientation: Orientation;
};

export function subtitle(kind: string, meta: Record<string, any>): string {
  switch (kind) {
    case "knowledge_base":
      return `${formatNumber(meta.documents)} docs · ${formatNumber(meta.chunks)} chunks`;
    case "database":
    case "warehouse":
    case "vector_db":
      return [meta.provider_label ?? meta.provider, meta.access_mode].filter(Boolean).join(" · ");
    case "integration":
      return meta.provider_label ?? meta.provider ?? "";
    case "document_source":
      return `${formatNumber(meta.documents)} uploaded docs`;
    case "pipeline":
      return meta.active_jobs ? `${meta.active_jobs} active jobs` : "idle";
    case "embedding":
      return `${meta.provider ?? ""}${meta.dim ? ` · ${meta.dim}d` : ""}`;
    case "vector_store":
      return `${formatNumber(meta.chunks)} vectors`;
    case "rag":
      return "hybrid · rerank · grounded";
    case "ai_model":
      return meta.model ? `${meta.kind} · ${meta.model}` : meta.note ?? meta.kind ?? "";
    case "agent":
      return `${(meta.tools ?? []).length} tools`;
    case "workspace":
      return `${meta.kb_count ?? 0} knowledge bases`;
    case "group":
      return `${meta.count} sources${meta.failed ? ` · ${meta.failed} failed` : ""}`;
    case "organization":
      return "organization";
    default:
      return "";
  }
}

/** Semantic zoom: hide the subtitle, then the label, as the map zooms out. */
const lodSelector = (s: { transform: [number, number, number] }) => (s.transform[2] < 0.36 ? 0 : s.transform[2] < 0.56 ? 1 : 2);

function BaseNode({ id, data, selected }: { id: string; data: MapNodeData; selected?: boolean }) {
  const lod = useStore(lodSelector);
  const Icon = NODE_ICON[data.kind] ?? Boxes;
  const family = NODE_FAMILY[data.kind] ?? "source";
  const tone = STATUS_TONE[data.status] ?? "neutral";
  const lg = data.kind === "organization";
  const vertical = data.orientation === "TB";
  const flashTone = data.flash === "error" ? "danger" : data.flash === "ok" ? "ok" : data.flash === "new" ? "accent" : null;
  const delay = `${-((hash(id) % 1000) / 1000) * 2.4}s`;
  const sub = subtitle(data.kind, data.meta);
  return (
    <div
      className={cn("cx-node", lg && "cx-node--lg", vertical && "is-tb", data.dimmed && "is-dimmed", selected && "is-selected", data.flash === "new" && "is-new")}
      style={{ "--tone": toneVar(tone), ...(flashTone ? { "--flash": toneVar(flashTone) } : {}) } as React.CSSProperties}
    >
      {vertical ? <Handle type="target" position={Position.Top} isConnectable={false} /> : null}
      <div className="cx-tile-wrap">
        {lg ? <span className="cx-aura" aria-hidden /> : null}
        {lg && data.animate ? (
          <>
            <span className="cx-ping" aria-hidden />
            <span className="cx-ping" aria-hidden />
          </>
        ) : null}
        <span className="cx-ring" data-status={data.status} aria-hidden />
        {data.status === "syncing" && data.animate ? <span className="cx-sweep" aria-hidden /> : null}
        {data.status === "failed" && data.animate ? <span className="cx-pulse" style={{ animationDelay: delay }} aria-hidden /> : null}
        {flashTone ? <span key={data.flash} className="cx-flash" aria-hidden /> : null}
        <div className={cn("cx-tile", `cx-tile--${family}`)}>
          <Icon />
        </div>
        {data.status === "failed" || data.status === "warning" ? (
          <span className="cx-badge" aria-hidden>
            !
          </span>
        ) : null}
        {!vertical ? (
          <>
            <Handle type="target" position={Position.Left} isConnectable={false} />
            <Handle type="source" position={Position.Right} isConnectable={false} />
          </>
        ) : null}
      </div>
      <div className="cx-label" style={{ opacity: data.showLabels && lod > 0 ? 1 : 0 }} title={data.label}>
        <div className="cx-label-name">{data.label}</div>
        <div className="cx-label-sub" style={{ opacity: lod > 1 ? 1 : 0 }}>
          {sub || " "}
        </div>
      </div>
      {vertical ? <Handle type="source" position={Position.Bottom} isConnectable={false} /> : null}
    </div>
  );
}

type P = NodeProps & { data: MapNodeData };

export const OrganizationNode = memo(function OrganizationNode({ id, data, selected }: P) {
  return <BaseNode id={id} data={data} selected={selected} />;
});
export const DatabaseNode = memo(function DatabaseNode({ id, data, selected }: P) {
  return <BaseNode id={id} data={data} selected={selected} />;
});
export const IntegrationNode = memo(function IntegrationNode({ id, data, selected }: P) {
  return <BaseNode id={id} data={data} selected={selected} />;
});
export const KnowledgeBaseNode = memo(function KnowledgeBaseNode({ id, data, selected }: P) {
  return <BaseNode id={id} data={data} selected={selected} />;
});
export const VectorStoreNode = memo(function VectorStoreNode({ id, data, selected }: P) {
  return <BaseNode id={id} data={data} selected={selected} />;
});
export const AIServiceNode = memo(function AIServiceNode({ id, data, selected }: P) {
  return <BaseNode id={id} data={data} selected={selected} />;
});

export const nodeTypes = {
  organization: OrganizationNode,
  workspace: KnowledgeBaseNode,
  database: DatabaseNode,
  warehouse: DatabaseNode,
  vector_db: VectorStoreNode,
  integration: IntegrationNode,
  document_source: IntegrationNode,
  group: DatabaseNode,
  pipeline: AIServiceNode,
  knowledge_base: KnowledgeBaseNode,
  embedding: AIServiceNode,
  vector_store: VectorStoreNode,
  rag: AIServiceNode,
  ai_model: AIServiceNode,
  agent: AIServiceNode,
};
