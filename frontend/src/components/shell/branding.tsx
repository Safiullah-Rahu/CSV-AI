"use client";

import { useEffect, useState } from "react";

import { useTheme } from "@/app/providers";
import { useAuth } from "@/lib/auth";
import { cn } from "@/lib/utils";

const HEX = /^#[0-9a-fA-F]{6}$/;

function hexToHsl(hex: string): [number, number, number] {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l * 100];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  const h = max === r ? (g - b) / d + (g < b ? 6 : 0) : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return [h * 60, s * 100, l * 100];
}

/**
 * Applies the organization's branding accent (Settings → Organization → Branding)
 * to the `--accent` token. Lightness is clamped per theme so text and buttons in
 * the accent colour stay legible on both the light and the dark surfaces.
 */
export function useOrgBranding() {
  const { org } = useAuth();
  const { resolved } = useTheme();
  const accent = org?.organization.settings?.branding?.accent as string | null | undefined;
  useEffect(() => {
    const root = document.documentElement;
    if (!accent || !HEX.test(accent)) {
      root.style.removeProperty("--accent");
      root.style.removeProperty("--ring");
      return;
    }
    const [h, s, l] = hexToHsl(accent);
    const lightness = resolved === "dark" ? Math.max(l, 62) : Math.min(l, 48);
    const value = `${h.toFixed(0)} ${s.toFixed(0)}% ${lightness.toFixed(0)}%`;
    root.style.setProperty("--accent", value);
    root.style.setProperty("--ring", value);
    return () => {
      root.style.removeProperty("--accent");
      root.style.removeProperty("--ring");
    };
  }, [accent, resolved]);
}

/** Square organization mark: the branding logo when set and loadable, else a monogram. */
export function OrgAvatar({ name, logo, className }: { name: string; logo?: string | null; className?: string }) {
  const [broken, setBroken] = useState<string | null>(null);
  const ini = name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0])
    .join("")
    .toUpperCase();
  if (logo && broken !== logo) {
    return (
      // eslint-disable-next-line @next/next/no-img-element -- external, admin-configured URL
      <img src={logo} alt="" referrerPolicy="no-referrer" onError={() => setBroken(logo)} className={cn("h-7 w-7 shrink-0 rounded-lg bg-surface object-contain ring-1 ring-border", className)} />
    );
  }
  return (
    <span className={cn("grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-fg text-[11px] font-semibold tracking-wide text-bg shadow-[inset_0_1px_0_hsl(0_0%_100%/0.15)]", className)}>
      {ini || "·"}
    </span>
  );
}
