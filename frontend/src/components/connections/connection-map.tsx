"use client";

import "@xyflow/react/dist/style.css";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Background, BackgroundVariant, type Edge, type Node, type NodeChange, ReactFlow, ReactFlowProvider, useNodesInitialized, useReactFlow, useUpdateNodeInternals } from "@xyflow/react";
import { List, Network, Workflow } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ConnectionFilters, MapControls, ScopeSelects, SearchFocus, ViewSwitch } from "@/components/connections/controls";
import { edgeTypes, type MapEdgeData } from "@/components/connections/edges";
import { clamp, freeRect, type Rect } from "@/components/connections/fit";
import { extent, layout, NODE_W, type Orientation, TILE, TILE_LG } from "@/components/connections/layout";
import { STATUS_LABEL, TYPE_LABEL } from "@/components/connections/meta";
import { nodeTypes, type MapNodeData } from "@/components/connections/nodes";
import { ConnectionDetailsPanel, ConnectionHealthPanel, ConnectionSummaryPanel, LiveActivityFeed } from "@/components/connections/panels";
import { CATEGORY_FILTERS, type GraphNode, type GraphResponse, SOURCE_TYPES, STATUS_FILTERS } from "@/components/connections/types";
import { ConnectionsListView, DataFlowView } from "@/components/connections/views";
import { Button } from "@/components/ui/button";
import { Dot, EmptyState } from "@/components/ui/primitives";
import { get } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { type ActivityEvent, useApi, useEventStream } from "@/lib/hooks";
import { cn, timeAgo } from "@/lib/utils";

type View = "map" | "list" | "flow";
type PanelKey = "summary" | "health" | "feed";
type XY = { x: number; y: number };

const VIEWS = [
  ["map", Network, "Live map"],
  ["list", List, "List"],
  ["flow", Workflow, "Data flow"],
] as const;

/** Width of the detail drawer plus its gutters (desktop). */
const DRAWER = 372 + 24;

function readPref(key: string, fallback: boolean): boolean {
  try {
    const v = localStorage.getItem(key);
    return v === null ? fallback : v === "1";
  } catch {
    return fallback;
  }
}
function writePref(key: string, v: boolean) {
  try {
    localStorage.setItem(key, v ? "1" : "0");
  } catch {
    /* per-session only */
  }
}

function nodeIdForEvent(e: ActivityEvent): string | null {
  if (!e.resource_id) return null;
  if (e.resource_type === "db_connection") return `db:${e.resource_id}`;
  if (e.resource_type === "integration") return `int:${e.resource_id}`;
  if (e.resource_type === "knowledge_base") return `kb:${e.resource_id}`;
  if (e.resource_type === "document" && e.payload?.kb_id) return `kb:${e.payload.kb_id}`;
  if (e.resource_type === "agent") return `agent:${e.resource_id}`;
  if (e.resource_type === "ai_provider") return `model:${e.resource_id}`;
  return null;
}

function LivePill({ connected, animate }: { connected: boolean; animate: boolean }) {
  return (
    <span
      className={cn(
        "inline-flex h-[22px] shrink-0 items-center gap-1.5 rounded-full px-2 font-mono text-[10px] font-medium uppercase tracking-[0.12em] ring-1 ring-inset",
        connected ? "bg-ok/10 text-ok ring-ok/20" : "bg-surface-2 text-subtle ring-border",
      )}
      title={connected ? "Receiving live events" : "Reconnecting to the event stream"}
    >
      <Dot tone={connected ? "ok" : "neutral"} live={connected && animate} />
      {connected ? "Live" : "Connecting"}
    </span>
  );
}

function MapInner() {
  const { orgId } = useAuth();
  const qc = useQueryClient();
  const rf = useReactFlow();
  const updateNodeInternals = useUpdateNodeInternals();
  const nodesInitialized = useNodesInitialized();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const [view, setView] = useState<View>("map");
  const [category, setCategory] = useState("all");
  const [statuses, setStatuses] = useState<string[]>([]);
  const [workspaceId, setWorkspaceId] = useState("");
  const [kbId, setKbId] = useState("");
  const [grouped, setGrouped] = useState(false);
  const [animate, setAnimateState] = useState(true);
  const [labels, setLabelsState] = useState(true);
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [events, setEvents] = useState<ActivityEvent[]>([]);
  const [flashes, setFlashes] = useState<Record<string, "ok" | "error" | "new">>({});
  const [connected, setConnected] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [orientation, setOrientation] = useState<Orientation>("LR");
  const [collapsed, setCollapsed] = useState<Record<PanelKey, boolean>>({ summary: false, health: false, feed: false });
  const [canvasW, setCanvasW] = useState(1200);
  const [ready, setReady] = useState(false);
  const [dragPos, setDragPos] = useState<Record<string, XY>>({});
  /** Measured node sizes fed back to React Flow (controlled nodes keep their measurements). */
  const [measured, setMeasured] = useState<Record<string, { width: number; height: number }>>({});
  const knownIds = useRef<Set<string>>(new Set());
  const refetchTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  /** True once the user pans/zooms; automatic refits then leave the viewport alone. */
  const userMoved = useRef(false);
  const readyRef = useRef(false);

  useEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    setAnimateState(readPref("rag.map.animate", !reduced));
    setLabelsState(readPref("rag.map.labels", true));
    // Panels start open on roomy canvases and collapsed on small/short ones; the user's choice sticks.
    const el = canvasRef.current;
    const w = el?.clientWidth ?? window.innerWidth;
    const h = el?.clientHeight ?? window.innerHeight;
    const small = w < 640;
    setCanvasW(w);
    setCollapsed({
      summary: readPref("rag.map.collapse.summary", small || h < 560),
      health: readPref("rag.map.collapse.health", small || h < 860),
      feed: readPref("rag.map.collapse.feed", small || h < 800),
    });
    const onFs = () => setFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener("fullscreenchange", onFs);
    return () => document.removeEventListener("fullscreenchange", onFs);
  }, []);
  const setAnimate = (v: boolean) => { setAnimateState(v); writePref("rag.map.animate", v); };
  const setLabels = (v: boolean) => { setLabelsState(v); writePref("rag.map.labels", v); };
  const togglePanel = (k: PanelKey) =>
    setCollapsed((c) => {
      writePref(`rag.map.collapse.${k}`, !c[k]);
      return { ...c, [k]: !c[k] };
    });

  const graph = useQuery({
    queryKey: ["connections-graph", orgId, workspaceId, kbId, grouped],
    queryFn: () => get<GraphResponse>("/connections/graph", { workspace_id: workspaceId || undefined, kb_id: kbId || undefined, group: grouped ? true : undefined }),
    enabled: Boolean(orgId),
    refetchInterval: 60_000,
  });
  const workspaces = useApi<{ id: string; name: string }[]>("/workspaces");
  const kbs = useApi<{ id: string; name: string }[]>("/knowledge-bases");
  const initialEvents = useApi<{ items: ActivityEvent[] }>("/events", { limit: 40 });
  useEffect(() => {
    if (initialEvents.data && events.length === 0) setEvents(initialEvents.data.items.filter((e) => e.type !== "job.progress"));
  }, [initialEvents.data]); // eslint-disable-line react-hooks/exhaustive-deps

  // Animate nodes that appear after the first load (newly connected resources).
  useEffect(() => {
    if (!graph.data) return;
    const ids = new Set(graph.data.nodes.map((n) => n.id));
    if (knownIds.current.size) {
      const fresh = [...ids].filter((id) => !knownIds.current.has(id));
      if (fresh.length) {
        setFlashes((f) => ({ ...f, ...Object.fromEntries(fresh.map((id) => [id, "new" as const])) }));
        setTimeout(() => setFlashes((f) => Object.fromEntries(Object.entries(f).filter(([k]) => !fresh.includes(k)))), 1500);
      }
    }
    knownIds.current = ids;
  }, [graph.data]);

  const onEvent = useCallback(
    (e: ActivityEvent) => {
      if (e.type === "job.progress") return;
      setEvents((prev) => (prev.some((p) => p.id === e.id) ? prev : [e, ...prev].slice(0, 60)));
      const nid = nodeIdForEvent(e);
      if (nid) {
        const tone = e.severity === "error" ? "error" : e.severity === "success" ? "ok" : null;
        if (tone) {
          setFlashes((f) => ({ ...f, [nid]: tone }));
          setTimeout(() => setFlashes((f) => { const { [nid]: _drop, ...rest } = f; return rest; }), 2600);
        }
      }
      if (refetchTimer.current) clearTimeout(refetchTimer.current);
      refetchTimer.current = setTimeout(() => {
        qc.invalidateQueries({ queryKey: ["connections-graph"] });
        qc.invalidateQueries({ queryKey: ["connections-health"] });
        qc.invalidateQueries({ queryKey: ["node-detail"] });
      }, 800);
    },
    [qc],
  );
  useEventStream(onEvent, true, setConnected);

  const nodes = useMemo(() => graph.data?.nodes ?? [], [graph.data]);
  const catFilter = CATEGORY_FILTERS.find((c) => c.key === category) ?? CATEGORY_FILTERS[0];
  const matches = useCallback(
    (n: GraphNode) => catFilter.match(n) && (statuses.length === 0 || statuses.some((s) => STATUS_FILTERS.find((f) => f.key === s)!.match(n))),
    [catFilter, statuses],
  );

  const positions = useMemo(() => layout(nodes, graph.data?.edges ?? [], orientation), [graph.data, orientation]); // eslint-disable-line react-hooks/exhaustive-deps
  const nodesRef = useRef<GraphNode[]>(nodes);
  nodesRef.current = nodes;
  const orientationRef = useRef(orientation);
  orientationRef.current = orientation;
  const filtering = category !== "all" || statuses.length > 0;
  const rfNodes: Node<MapNodeData>[] = useMemo(
    () =>
      nodes.map((n) => ({
        id: n.id,
        type: n.type in nodeTypes ? n.type : "database",
        position: dragPos[n.id] ?? positions[n.id] ?? { x: 0, y: 0 },
        measured: measured[n.id],
        data: { label: n.label, kind: n.type, status: n.status, meta: n.data, showLabels: labels, animate, dimmed: filtering && !matches(n) && n.type !== "organization", flash: flashes[n.id] ?? null, orientation },
        draggable: true,
        selectable: true,
        selected: selected?.id === n.id,
        ariaLabel: `${n.label}, ${TYPE_LABEL[n.type] ?? n.type.replace(/_/g, " ")}, ${STATUS_LABEL[n.status] ?? n.status}`,
      })),
    [nodes, positions, dragPos, measured, labels, animate, filtering, matches, flashes, orientation, selected?.id],
  );
  const dimmedIds = useMemo(() => new Set(rfNodes.filter((n) => n.data.dimmed).map((n) => n.id)), [rfNodes]);
  const rfEdges: Edge<MapEdgeData>[] = useMemo(
    () =>
      (graph.data?.edges ?? []).map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        type: "animated",
        data: {
          status: e.status,
          kind: e.kind,
          animate: animate && (graph.data!.edges.length <= 400 || e.status === "active"),
          dimmed: dimmedIds.has(e.source) || dimmedIds.has(e.target),
          highlight: Boolean(selected) && (e.source === selected!.id || e.target === selected!.id),
        },
      })),
    [graph.data, animate, dimmedIds, selected],
  );

  /** Canvas size and the rectangles covered by floating panels (canvas coordinates). */
  const measure = useCallback((): { W: number; H: number; obstacles: Rect[] } | null => {
    const el = canvasRef.current;
    if (!el) return null;
    const cr = el.getBoundingClientRect();
    const obstacles: Rect[] = [];
    el.querySelectorAll<HTMLElement>("[data-map-obstacle]").forEach((o) => {
      const r = o.getBoundingClientRect();
      if (r.width > 0 && r.height > 0) obstacles.push({ x: r.left - cr.left, y: r.top - cr.top, w: r.width, h: r.height });
    });
    return { W: cr.width, H: cr.height, obstacles };
  }, []);

  /**
   * Fit the graph into the largest area no panel covers. Automatic fits keep a
   * legibility floor (labels stay readable; very large graphs overflow and can
   * be panned); "Fit all nodes" goes down to the minimum zoom.
   */
  const fit = useCallback(
    (opts: { duration?: number; all?: boolean } = {}) => {
      const m = measure();
      const all = rf.getNodes();
      if (!m || !all.length) return;
      const b = rf.getNodesBounds(all);
      const r = freeRect(m.W, m.H, m.obstacles, b.width, b.height);
      const floor = opts.all ? 0.15 : m.W < 640 ? 0.6 : 0.5;
      const zoom = clamp(Math.min(r.w / Math.max(b.width, 1), r.h / Math.max(b.height, 1)), floor, 1.15);
      const gw = b.width * zoom;
      const gh = b.height * zoom;
      // Too big even at the floor: along the data path, start at its beginning; across it, stay centred.
      const lr = orientationRef.current === "LR";
      const x = (gw <= r.w || !lr ? r.x + (r.w - gw) / 2 : r.x) - b.x * zoom;
      const y = (gh <= r.h || lr ? r.y + (r.h - gh) / 2 : r.y) - b.y * zoom;
      rf.setViewport({ x, y, zoom }, { duration: opts.duration ?? 400 });
      userMoved.current = false;
    },
    [rf, measure],
  );

  /** Left → right on wide canvases, top → bottom when that draws the graph clearly larger. */
  const pickOrientation = useCallback(
    (list: GraphNode[], current: Orientation): Orientation => {
      const m = measure();
      if (!m || !list.length) return current;
      const scale = (o: Orientation) => {
        const e = extent(list, o);
        // Canvas shape only: opening or collapsing a panel must never flip the layout.
        return freeRect(m.W, m.H, [], e.w, e.h, 24).scale;
      };
      const cur = scale(current);
      const other: Orientation = current === "LR" ? "TB" : "LR";
      return scale(other) > cur * 1.12 ? other : current;
    },
    [measure],
  );

  // Refit from scratch whenever the graph query itself changes (workspace, KB, grouping) or the view returns to the map.
  const graphKey = `${view}|${workspaceId}|${kbId}|${grouped}`;
  useEffect(() => {
    readyRef.current = false;
    setReady(false);
  }, [graphKey]);

  // First fit once React Flow has measured the nodes.
  useEffect(() => {
    if (view !== "map" || !graph.data || readyRef.current) return;
    if (!nodes.length) {
      readyRef.current = true;
      setReady(true);
      return;
    }
    if (!nodesInitialized) return;
    const o = pickOrientation(nodes, orientation);
    if (o !== orientation) {
      setDragPos({});
      setOrientation(o);
      return;
    }
    const id = requestAnimationFrame(() => {
      fit({ duration: 0 });
      readyRef.current = true;
      setReady(true);
    });
    return () => cancelAnimationFrame(id);
  }, [view, graph.data, nodes, nodesInitialized, orientation, pickOrientation, fit]);

  // Safety net: very large graphs render only visible nodes, so "initialized" may never flip.
  useEffect(() => {
    if (view !== "map" || !graph.data) return;
    const t = setTimeout(() => {
      if (readyRef.current) return;
      fit({ duration: 0 });
      readyRef.current = true;
      setReady(true);
    }, 1200);
    return () => clearTimeout(t);
  }, [view, graph.data, fit]);

  // Orientation changed: handles moved, so re-measure them and refit.
  useEffect(() => {
    if (!nodesRef.current.length) return;
    updateNodeInternals(nodesRef.current.map((n) => n.id));
    if (!readyRef.current) return;
    const id = requestAnimationFrame(() => fit({ duration: 350 }));
    return () => cancelAnimationFrame(id);
  }, [orientation]); // eslint-disable-line react-hooks/exhaustive-deps

  /** With the drawer open, pan (never zoom) just enough to keep the selected node beside it. */
  const keepBesideDrawer = useCallback(
    (id: string) => {
      const el = canvasRef.current;
      const n = rf.getInternalNode(id);
      if (!el || !n || el.clientWidth < 640) return; // on phones the drawer covers the map
      const { x, y, zoom } = rf.getViewport();
      const right = x + (n.internals.positionAbsolute.x + (n.measured.width ?? NODE_W)) * zoom;
      const limit = el.clientWidth - DRAWER - 16;
      if (right > limit) rf.setViewport({ x: x - (right - limit), y, zoom }, { duration: 360 });
    },
    [rf],
  );
  const selectedIdRef = useRef<string | null>(null);
  selectedIdRef.current = selected?.id ?? null;
  const drawerOpen = Boolean(selected);

  // Canvas resized or a panel opened/collapsed: re-pick orientation, refit unless the user has moved the map.
  useEffect(() => {
    if (view !== "map") return;
    const el = canvasRef.current;
    if (!el) return;
    let t: ReturnType<typeof setTimeout> | undefined;
    const ro = new ResizeObserver(() => {
      if (t) clearTimeout(t);
      t = setTimeout(() => {
        setCanvasW(el.clientWidth);
        if (!readyRef.current || userMoved.current) return;
        if (drawerOpen) {
          if (selectedIdRef.current) keepBesideDrawer(selectedIdRef.current);
          return;
        }
        const o = pickOrientation(nodesRef.current, orientationRef.current);
        if (o !== orientationRef.current) {
          setDragPos({});
          setOrientation(o);
          return;
        }
        fit({ duration: 320 });
      }, 140);
    });
    ro.observe(el);
    el.querySelectorAll<HTMLElement>("[data-map-obstacle]").forEach((o) => ro.observe(o));
    return () => {
      ro.disconnect();
      if (t) clearTimeout(t);
    };
  }, [view, drawerOpen, Boolean(graph.data), pickOrientation, fit, keepBesideDrawer]); // eslint-disable-line react-hooks/exhaustive-deps

  // Escape closes the detail drawer.
  useEffect(() => {
    if (!selected) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !document.querySelector("[role=dialog]")) setSelected(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selected]);

  const onNodesChange = useCallback((changes: NodeChange<Node<MapNodeData>>[]) => {
    const moved: Record<string, XY> = {};
    const dims: Record<string, { width: number; height: number }> = {};
    for (const c of changes) {
      if (c.type === "position" && c.position) moved[c.id] = c.position;
      if (c.type === "dimensions" && c.dimensions) dims[c.id] = c.dimensions;
    }
    if (Object.keys(moved).length) setDragPos((p) => ({ ...p, ...moved }));
    if (Object.keys(dims).length) setMeasured((p) => ({ ...p, ...dims }));
  }, []);

  const focus = (n: GraphNode) => {
    const p = dragPos[n.id] ?? positions[n.id];
    const el = canvasRef.current;
    if (p && el && view === "map") {
      const zoom = 1.4;
      const cx = p.x + NODE_W / 2;
      const cy = p.y + (n.type === "organization" ? TILE_LG : TILE) / 2 + 18;
      const W = el.clientWidth;
      const free = W >= 640 ? W - DRAWER : W; // the drawer opens on the right
      rf.setViewport({ x: free / 2 - cx * zoom, y: el.clientHeight / 2 - cy * zoom, zoom }, { duration: 600 });
      userMoved.current = true;
    }
    setSelected(n);
  };
  const resetFilters = () => { setCategory("all"); setStatuses([]); setWorkspaceId(""); setKbId(""); };
  const orgNode = nodes.find((n) => n.type === "organization");
  const sourceNodes = nodes.filter((n) => SOURCE_TYPES.has(n.type) && n.type !== "group" && matches(n));
  const kbNames = Object.fromEntries((kbs.data ?? []).map((k) => [k.id, k.name]));
  const nodeIds = useMemo(() => new Set(nodes.map((n) => n.id)), [nodes]);
  const compact = canvasW < 640 || canvasW - (drawerOpen ? DRAWER : 0) < 760;
  const rightInset = drawerOpen ? "right-3 sm:right-[396px]" : "right-3 sm:right-4";
  const scope = { workspaces: workspaces.data ?? [], workspaceId, setWorkspaceId, kbs: kbs.data ?? [], kbId, setKbId };

  return (
    <div ref={containerRef} className="cx-root relative flex h-full min-h-0 flex-col overflow-hidden bg-bg text-fg">
      {/* toolbar */}
      <header className="relative z-20 border-b bg-bg">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2 px-4 pb-2 pt-3 sm:px-5">
          <div className="flex min-w-0 items-center gap-2.5">
            <h1 className="truncate text-[16px] font-semibold tracking-[-0.02em]">Connections</h1>
            <LivePill connected={connected} animate={animate} />
          </div>
          <div className="ml-auto sm:ml-1">
            <ViewSwitch value={view} onChange={setView} options={VIEWS} />
          </div>
          {view === "map" ? <SearchFocus nodes={nodes} onFocus={focus} className="order-last w-full sm:order-none sm:w-56 xl:w-64" /> : null}
          <ScopeSelects {...scope} className="ml-auto hidden min-[1400px]:flex" />
        </div>
        <div className="px-4 pb-2.5 sm:px-5">
          <ConnectionFilters nodes={nodes} category={category} setCategory={setCategory} statuses={statuses} setStatuses={setStatuses}>
            <span className="mx-1 h-4 w-px shrink-0 bg-border min-[1400px]:hidden" aria-hidden />
            <ScopeSelects {...scope} className="min-[1400px]:hidden [&_select]:!h-7 [&_select]:text-[12px]" />
          </ConnectionFilters>
        </div>
      </header>

      <div ref={canvasRef} className="relative min-h-0 flex-1 overflow-hidden">
        {view === "map" ? (
          <div className="cx-stage absolute inset-0">
            <div className="cx-backdrop pointer-events-none absolute inset-0" aria-hidden />
            <ReactFlow
              nodes={rfNodes}
              edges={rfEdges}
              nodeTypes={nodeTypes}
              edgeTypes={edgeTypes}
              onNodesChange={onNodesChange}
              onNodeClick={(_, n) => setSelected(nodes.find((x) => x.id === n.id) ?? null)}
              onPaneClick={() => setSelected(null)}
              onMoveStart={(e) => { if (e) userMoved.current = true; }}
              nodesConnectable={false}
              minZoom={0.15}
              maxZoom={2.5}
              onlyRenderVisibleElements={nodes.length > 150}
              proOptions={{ hideAttribution: true }}
              className={cn("cx-map !bg-transparent transition-opacity duration-500", ready ? "opacity-100" : "opacity-0")}
            >
              <Background id="cx-dots" variant={BackgroundVariant.Dots} gap={22} size={1.1} color="var(--cx-dot)" />
              <Background id="cx-grid" variant={BackgroundVariant.Lines} gap={176} lineWidth={1} color="var(--cx-grid)" />
            </ReactFlow>

            {graph.isLoading ? (
              <div className="absolute inset-0 grid place-items-center" role="status">
                <div className="flex flex-col items-center gap-4">
                  <span className="cx-loader" aria-hidden />
                  <span className="label">Mapping your data ecosystem</span>
                </div>
              </div>
            ) : null}
            {graph.isError && !graph.data ? (
              <div className="absolute inset-0 z-[5] grid place-items-center p-6">
                <div className="cx-glass rounded-2xl">
                  <EmptyState title="The map could not be loaded" description="The connection graph did not respond. Your data is unaffected; try again in a moment." action={<Button variant="secondary" size="sm" onClick={() => graph.refetch()}>Try again</Button>} />
                </div>
              </div>
            ) : null}
            {graph.data && nodes.length <= 6 && !drawerOpen ? (
              <div className="pointer-events-none absolute inset-x-0 bottom-4 z-[5] hidden justify-center xl:flex">
                <div className="cx-glass pointer-events-auto flex max-w-[calc(100%-720px)] items-center gap-3 rounded-xl px-4 py-2.5 text-[12.5px] text-muted">
                  <span>Every node here is a real resource you can access. Connect databases and integrations to grow the map.</span>
                  <Link href="/database-connections" className="shrink-0 font-medium text-accent hover:underline">
                    Connect a source
                  </Link>
                </div>
              </div>
            ) : null}

            {/* instrument readout */}
            {graph.data && !drawerOpen ? (
              <div data-map-obstacle className="pointer-events-none absolute bottom-[26px] left-1/2 z-[4] hidden -translate-x-1/2 items-center gap-2.5 whitespace-nowrap font-mono text-[10.5px] uppercase tracking-[0.12em] text-subtle lg:flex" aria-hidden>
                <span><span className="num text-muted">{nodes.length}</span> nodes</span>
                <span className="h-2.5 w-px bg-border-strong" />
                <span><span className="num text-muted">{graph.data.edges.length}</span> links</span>
                <span className="h-2.5 w-px bg-border-strong" />
                <span>updated {timeAgo(graph.data.generated_at)}</span>
              </div>
            ) : null}

            {/* floating panels: each one is an obstacle the fit works around */}
            <div data-map-obstacle className={cn("absolute left-3 z-10 max-w-[calc(100%-24px)] sm:left-4 sm:top-4", canvasW < 640 && !collapsed.summary ? "top-[60px]" : "top-3")}>
              <ConnectionSummaryPanel s={graph.data?.summary} collapsed={collapsed.summary} onToggle={() => togglePanel("summary")} />
            </div>
            <div data-map-obstacle className={cn("absolute top-3 z-10 sm:top-4", rightInset)}>
              <MapControls
                compact={compact}
                onZoomIn={() => { userMoved.current = true; rf.zoomIn({ duration: 200 }); }}
                onZoomOut={() => { userMoved.current = true; rf.zoomOut({ duration: 200 }); }}
                onFit={() => fit({ all: true })}
                onReset={() => { resetFilters(); setSelected(null); setDragPos({}); requestAnimationFrame(() => fit()); }}
                onCenterOrg={() => orgNode && focus(orgNode)}
                animate={animate}
                setAnimate={setAnimate}
                labels={labels}
                setLabels={setLabels}
                grouped={grouped}
                setGrouped={setGrouped}
                fullscreen={fullscreen}
                onFullscreen={() => (document.fullscreenElement ? document.exitFullscreen() : containerRef.current?.requestFullscreen())}
              />
            </div>
            <div data-map-obstacle className={cn("absolute bottom-3 left-3 z-10 sm:bottom-4 sm:left-4", drawerOpen && canvasW < 1120 ? "hidden" : "hidden sm:block")}>
              <ConnectionHealthPanel orgKey={orgId} collapsed={collapsed.health} onToggle={() => togglePanel("health")} />
            </div>
            <div data-map-obstacle className={cn("absolute bottom-3 left-3 z-10 flex justify-end sm:bottom-4 sm:left-auto", rightInset)}>
              <LiveActivityFeed
                events={events}
                connected={connected}
                animate={animate}
                collapsed={collapsed.feed}
                onToggle={() => togglePanel("feed")}
                canSelect={(e) => { const nid = nodeIdForEvent(e); return Boolean(nid && nodeIds.has(nid)); }}
                onSelect={(e) => { const nid = nodeIdForEvent(e); const n = nodes.find((x) => x.id === nid); if (n) focus(n); }}
              />
            </div>
          </div>
        ) : view === "list" ? (
          <div className="h-full overflow-y-auto">
            <ConnectionsListView nodes={sourceNodes} kbNames={kbNames} onSelect={(n) => setSelected(n)} selectedId={selected?.id} filtering={filtering || Boolean(workspaceId || kbId)} onClearFilters={resetFilters} />
          </div>
        ) : (
          <DataFlowView orgKey={orgId} kbId={kbId} animate={animate} />
        )}
        {selected ? (
          <div data-map-obstacle className="absolute inset-2 z-30 sm:inset-auto sm:bottom-3 sm:right-3 sm:top-3">
            <ConnectionDetailsPanel node={selected} onClose={() => setSelected(null)} onChanged={() => qc.invalidateQueries({ queryKey: ["connections-graph"] })} />
          </div>
        ) : null}
      </div>
    </div>
  );
}

export function ConnectionMap() {
  return (
    <ReactFlowProvider>
      <MapInner />
    </ReactFlowProvider>
  );
}
