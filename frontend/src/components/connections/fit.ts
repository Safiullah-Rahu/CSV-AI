/**
 * Viewport fitting around floating panels.
 *
 * The map canvas is partly covered by glass panels (summary, health, live
 * activity, controls, detail drawer). Instead of reserving a fixed gutter, we
 * look for the free rectangle (one that touches no panel) in which the graph's
 * bounding box can be drawn largest, and fit the graph there.
 */
export type Rect = { x: number; y: number; w: number; h: number };

const overlaps = (a: Rect, b: Rect) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;

/**
 * Largest-scale free rectangle for content of size `cw × ch` inside a
 * `W × H` canvas with `obstacles`. Candidate edges come from the canvas
 * padding and every obstacle edge, so the search is exhaustive but tiny
 * (≤ 6 obstacles → ≤ 7⁴ candidates).
 */
export function freeRect(W: number, H: number, obstacles: Rect[], cw: number, ch: number, pad = 16, gap = 14): Rect & { scale: number } {
  const obs = obstacles.filter((o) => o.w > 0 && o.h > 0 && o.x < W && o.y < H && o.x + o.w > 0 && o.y + o.h > 0);
  const xs1 = [pad, ...obs.map((o) => o.x + o.w + gap)];
  const xs2 = [W - pad, ...obs.map((o) => o.x - gap)];
  const ys1 = [pad, ...obs.map((o) => o.y + o.h + gap)];
  const ys2 = [H - pad, ...obs.map((o) => o.y - gap)];
  const cwS = Math.max(cw, 1);
  const chS = Math.max(ch, 1);
  let best: (Rect & { scale: number }) | null = null;
  let bestScore = -Infinity;
  for (const x1 of xs1) {
    if (x1 < pad - 0.5) continue;
    for (const x2 of xs2) {
      if (x2 > W - pad + 0.5 || x2 - x1 < 96) continue;
      for (const y1 of ys1) {
        if (y1 < pad - 0.5) continue;
        for (const y2 of ys2) {
          if (y2 > H - pad + 0.5 || y2 - y1 < 96) continue;
          const r = { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
          if (obs.some((o) => overlaps(r, o))) continue;
          const scale = Math.min(r.w / cwS, r.h / chS);
          // Prefer the biggest scale; break ties toward larger, more central areas.
          const centre = Math.hypot(r.x + r.w / 2 - W / 2, r.y + r.h / 2 - H / 2) / Math.max(W, H);
          const score = scale + (r.w * r.h) / (W * H) * 1e-3 - centre * 1e-3;
          if (score > bestScore) {
            bestScore = score;
            best = { ...r, scale };
          }
        }
      }
    }
  }
  if (best) return best;
  const r = { x: pad, y: pad, w: Math.max(1, W - pad * 2), h: Math.max(1, H - pad * 2) };
  return { ...r, scale: Math.min(r.w / cwS, r.h / chS) };
}

export function clamp(v: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, v));
}
