"use client";

import { useEffect, useSyncExternalStore } from "react";

/*
 * Lets a detail page name the last breadcrumb ("Finance" instead of "Details").
 * One label at a time: the page that mounted last owns it and clears it on unmount.
 */
let label: string | null = null;
const listeners = new Set<() => void>();

function emit() {
  listeners.forEach((fn) => fn());
}

export function useBreadcrumbLabel(): string | null {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => label,
    () => null,
  );
}

export function usePageCrumb(text: string | null | undefined) {
  useEffect(() => {
    label = text || null;
    emit();
    return () => {
      label = null;
      emit();
    };
  }, [text]);
}
