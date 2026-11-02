"use client";

import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

import { get, streamSSE, type SSEEvent } from "@/lib/api";
import { useAuth } from "@/lib/auth";

/** GET helper scoped to the active organization (cache key includes the org). */
export function useApi<T>(path: string | null, query?: Record<string, any>, opts?: Partial<UseQueryOptions<T>>) {
  const { orgId } = useAuth();
  return useQuery<T>({
    queryKey: [path, orgId, query],
    queryFn: () => get<T>(path as string, query),
    enabled: Boolean(path && orgId) && (opts?.enabled ?? true),
    ...opts,
  });
}

export type ActivityEvent = {
  id: number;
  type: string;
  resource_type: string | null;
  resource_id: string | null;
  actor_user_id: string | null;
  severity: string;
  message: string;
  payload: Record<string, any>;
  created_at: string;
};

/** Live organization events over SSE with automatic reconnect + resume. */
export function useEventStream(onEvent: (ev: ActivityEvent) => void, enabled = true, onStatus?: (connected: boolean) => void) {
  const { orgId } = useAuth();
  const handler = useRef(onEvent);
  handler.current = onEvent;
  const status = useRef(onStatus);
  status.current = onStatus;
  useEffect(() => {
    if (!orgId || !enabled) return;
    let stopped = false;
    let lastId: string | undefined;
    let controller = new AbortController();
    let delay = 1000;
    const run = async () => {
      while (!stopped) {
        controller = new AbortController();
        try {
          await streamSSE("/events/stream", {
            signal: controller.signal,
            lastEventId: lastId,
            onOpen: () => status.current?.(true),
            onEvent: (e: SSEEvent) => {
              if (e.id) lastId = e.id;
              if (e.data && typeof e.data === "object" && "type" in e.data) handler.current(e.data as ActivityEvent);
              delay = 1000;
            },
          });
        } catch {
          /* reconnect below */
        }
        status.current?.(false);
        if (stopped) break;
        await new Promise((r) => setTimeout(r, delay));
        delay = Math.min(delay * 2, 15000);
      }
    };
    run();
    return () => {
      stopped = true;
      controller.abort();
    };
  }, [orgId, enabled]);
}
