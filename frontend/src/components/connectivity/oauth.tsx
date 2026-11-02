"use client";

import { useMutation } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

import { post } from "@/lib/api";

import { CodeBlock, Notice } from "./shared";
import type { IntegrationCatalogItem, IntegrationOAuthApp } from "./types";

/** True when this provider/mode authenticates through the vendor's OAuth consent screen. */
export function usesOAuth(item: Pick<IntegrationCatalogItem, "auth_modes" | "oauth"> | null | undefined, mode?: string | null): boolean {
  if (!item?.oauth) return false;
  const modes = item.auth_modes ?? [];
  return (mode ?? modes[0]) === "oauth";
}

/** OAuth-only provider whose app is not configured on this deployment: it cannot be connected at all. */
export function blockedByOAuthSetup(item: IntegrationCatalogItem): boolean {
  const modes = item.auth_modes ?? [];
  return Boolean(item.oauth && !item.oauth.configured && modes.length === 1 && modes[0] === "oauth");
}

/**
 * Starts the authorization-code flow for an existing integration and leaves the app for the
 * vendor's consent screen. The backend returns to /integrations?oauth=connected|error.
 */
export function useOAuthConnect() {
  return useMutation({
    mutationFn: (integrationId: string) => post<{ authorization_url: string; expires_in: number }>(`/integrations/${integrationId}/oauth/start`),
    onSuccess: (r) => {
      window.location.assign(r.authorization_url);
    },
  });
}

/** Reads `?oauth=connected|error&integration=…&message=…` once, then strips it from the URL. */
export function useOAuthReturn(onResult: (r: { status: "connected" | "error"; integration: string | null; message: string | null }) => void) {
  const handled = useRef(false);
  useEffect(() => {
    if (handled.current) return;
    const params = new URLSearchParams(window.location.search);
    const status = params.get("oauth");
    if (status !== "connected" && status !== "error") return;
    handled.current = true;
    onResult({ status, integration: params.get("integration"), message: params.get("message") });
    ["oauth", "integration", "message"].forEach((k) => params.delete(k));
    const rest = params.toString();
    window.history.replaceState(window.history.state, "", `${window.location.pathname}${rest ? `?${rest}` : ""}`);
  }, [onResult]);
}

/** What an operator must configure before this vendor's OAuth connection can be used. */
export function OAuthSetupNotice({ app, label, compact }: { app: IntegrationOAuthApp; label: string; compact?: boolean }) {
  const required = app.env.filter((e) => !e.includes("optional"));
  const optional = app.env.filter((e) => e.includes("optional"));
  return (
    <Notice tone="warn" title={`${app.vendor} OAuth app not configured on this deployment`}>
      <p>
        {label} connects through {app.vendor}&rsquo;s consent screen, which needs a platform OAuth app. A platform operator sets these environment variables on the API and worker, then restarts them:
      </p>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {required.map((e) => (
          <code key={e} className="rounded-md border bg-surface px-1.5 py-0.5 font-mono text-[11.5px] text-fg">
            {e}
          </code>
        ))}
        {optional.map((e) => (
          <code key={e} className="rounded-md border border-dashed px-1.5 py-0.5 font-mono text-[11.5px] text-muted">
            {e}
          </code>
        ))}
      </div>
      {compact ? null : (
        <div className="mt-3 space-y-2">
          <p className="text-[12px]">
            Register this redirect URI{app.console ? <> in {app.console}</> : null}
            {app.scopes.length ? (
              <>
                {" "}
                with the scopes <span className="font-mono text-[11.5px]">{app.scopes.join(" ")}</span>
              </>
            ) : null}
            :
          </p>
          <CodeBlock copy label="Redirect URI">
            {app.redirect_uri}
          </CodeBlock>
        </div>
      )}
    </Notice>
  );
}
