import type { Citation } from "@/components/chat/types";

/** Human-readable locator pieces for a citation (page / sheet / rows / section ...). */
export function locatorParts(loc: Record<string, any> | null | undefined): string[] {
  if (!loc) return [];
  const out: string[] = [];
  if (loc.page) out.push(`p. ${loc.page}${loc.page_end ? `–${loc.page_end}` : ""}`);
  if (loc.slide) out.push(`slide ${loc.slide}`);
  if (loc.sheet) out.push(`sheet ${loc.sheet}`);
  if (loc.row_start) out.push(`rows ${loc.row_start}–${loc.row_end}`);
  if (loc.line_start) out.push(`lines ${loc.line_start}–${loc.line_end}`);
  if (loc.table) out.push(`table ${loc.table}`);
  return out;
}

/** The deepest section heading, e.g. "Working hours" from ["Handbook", "Working hours"]. */
export function sectionLeaf(loc: Record<string, any> | null | undefined): string | null {
  const s = loc?.section;
  return Array.isArray(s) && s.length ? String(s[s.length - 1]) : null;
}

/** Compact one-line locator used in chips and lists. */
export function shortLocator(c: Citation): string {
  const parts = locatorParts(c.locator);
  const leaf = sectionLeaf(c.locator);
  if (leaf) parts.push(`§ ${leaf}`);
  return parts.join(" · ");
}

/** 76 ms · 4.2 s · 1m 54s */
export function formatDuration(ms: number | null | undefined): string | null {
  if (ms == null || Number.isNaN(ms)) return null;
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  const total = Math.round(ms / 1000);
  return `${Math.floor(total / 60)}m ${String(total % 60).padStart(2, "0")}s`;
}

const DAY = 86_400_000;

/** Bucket rows by how recently they were updated: Today, Yesterday, Previous 7 days ... */
export function groupByRecency<T extends { updated_at: string }>(rows: T[]): { label: string; items: T[] }[] {
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const buckets: { label: string; test: (t: number) => boolean }[] = [
    { label: "Today", test: (t) => t >= startOfToday },
    { label: "Yesterday", test: (t) => t >= startOfToday - DAY },
    { label: "Previous 7 days", test: (t) => t >= startOfToday - 7 * DAY },
    { label: "Previous 30 days", test: (t) => t >= startOfToday - 30 * DAY },
    { label: "Older", test: () => true },
  ];
  const out: { label: string; items: T[] }[] = [];
  for (const row of rows) {
    const t = new Date(row.updated_at).getTime();
    const b = buckets.find((x) => x.test(t)) ?? buckets[buckets.length - 1];
    let group = out.find((g) => g.label === b.label);
    if (!group) {
      group = { label: b.label, items: [] };
      out.push(group);
    }
    group.items.push(row);
  }
  return out;
}

/** Short relative time for dense lists: "now", "12m", "3h", "2d", or a date. */
export function shortAgo(iso: string | null | undefined): string {
  if (!iso) return "";
  const diff = (Date.now() - new Date(iso).getTime()) / 1000;
  if (diff < 60) return "now";
  if (diff < 3600) return `${Math.round(diff / 60)}m`;
  if (diff < 86400) return `${Math.round(diff / 3600)}h`;
  if (diff < 86400 * 7) return `${Math.round(diff / 86400)}d`;
  return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}
