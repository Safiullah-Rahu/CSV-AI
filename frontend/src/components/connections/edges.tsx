"use client";

import { BaseEdge, getBezierPath, type EdgeProps } from "@xyflow/react";
import { memo } from "react";

import { hash } from "@/components/connections/meta";

export type MapEdgeData = { status: string; kind: string; animate: boolean; dimmed: boolean; highlight?: boolean };

/**
 * Hairline bezier edge. Iris particles travel along the path only for links
 * whose real state is healthy (one, slow) or actively synchronising (three,
 * fast, on an iris line). Failed links render red and dashed; paused,
 * disabled and idle links are muted and dashed; live-query links are dotted.
 * Edges touching the selected node light up in iris.
 */
export const AnimatedEdge = memo(function AnimatedEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, data }: EdgeProps & { data?: MapEdgeData }) {
  const [path] = getBezierPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, curvature: 0.32 });
  const status = data?.status ?? "healthy";
  const dimmed = Boolean(data?.dimmed);
  const hl = Boolean(data?.highlight) && !dimmed;
  const failed = status === "failed";
  const active = status === "active";
  const muted = status === "disabled" || status === "idle" || status === "paused";
  const live = Boolean(data?.animate) && !dimmed && (status === "healthy" || active);
  const dash = failed || muted ? "5 5" : data?.kind === "live_query" ? "1.5 5" : undefined;
  const stroke = failed
    ? "hsl(var(--danger) / 0.75)"
    : active || hl
      ? "hsl(var(--accent) / 0.8)"
      : muted
        ? "hsl(var(--fg) / 0.16)"
        : "hsl(var(--fg) / 0.26)";
  const glow = failed ? "var(--danger)" : "var(--accent)";
  const particles = active ? 3 : 1;
  const duration = active ? 1.8 : 4.6;
  const phase = ((hash(id) % 1000) / 1000) * duration;
  return (
    <g style={{ opacity: dimmed ? 0.07 : 1, transition: "opacity 300ms" }}>
      {active || hl || failed ? <path d={path} fill="none" style={{ stroke: `hsl(${glow} / 0.13)`, strokeWidth: 7, strokeLinecap: "round" }} /> : null}
      <BaseEdge id={id} path={path} style={{ stroke, strokeWidth: hl ? 1.75 : active ? 1.5 : 1.25, strokeDasharray: dash, strokeLinecap: "round" }} />
      {live
        ? Array.from({ length: particles }).map((_, i) => (
            <g key={i}>
              <circle r={active ? 6 : 5} style={{ fill: "hsl(var(--accent) / 0.16)" }} />
              <circle r={active ? 2.5 : 2.1} style={{ fill: "hsl(var(--accent))" }} />
              {/* negative begin = start mid-path, so particles never move in lockstep */}
              <animateMotion dur={`${duration}s`} repeatCount="indefinite" begin={`${-(phase + (i * duration) / particles).toFixed(2)}s`} path={path} />
            </g>
          ))
        : null}
    </g>
  );
});

export const edgeTypes = { animated: AnimatedEdge };
