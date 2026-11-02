"use client";

import { MutationCache, QueryCache, QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { Toaster, toast } from "sonner";

import { ApiError, errorMessage } from "@/lib/api";
import { AuthProvider } from "@/lib/auth";

type Theme = "light" | "dark" | "system";
const ThemeContext = createContext<{ theme: Theme; resolved: "light" | "dark"; setTheme: (t: Theme) => void }>({ theme: "system", resolved: "light", setTheme: () => {} });

function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<Theme>("system");
  const [resolved, setResolved] = useState<"light" | "dark">("light");
  useEffect(() => {
    let stored: Theme = "system";
    try {
      stored = (localStorage.getItem("rag.theme") as Theme) || "system";
    } catch {
      /* default */
    }
    setThemeState(stored);
  }, []);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const apply = () => {
      const r = theme === "system" ? (mq.matches ? "dark" : "light") : theme;
      setResolved(r);
      document.documentElement.classList.toggle("dark", r === "dark");
    };
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, [theme]);
  const setTheme = useCallback((t: Theme) => {
    setThemeState(t);
    try {
      localStorage.setItem("rag.theme", t);
    } catch {
      /* not persisted */
    }
  }, []);
  return <ThemeContext.Provider value={{ theme, resolved, setTheme }}>{children}</ThemeContext.Provider>;
}

export const useTheme = () => useContext(ThemeContext);

export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            retry: (count, err) => !(err instanceof ApiError && err.status < 500) && count < 2,
            refetchOnWindowFocus: false,
            staleTime: 15_000,
          },
        },
        queryCache: new QueryCache({
          onError: (err, query) => {
            if (query.meta?.silent) return;
            if (err instanceof ApiError && (err.status === 401 || err.status === 404)) return;
            toast.error(errorMessage(err));
          },
        }),
        mutationCache: new MutationCache({
          onError: (err, _v, _c, mutation) => {
            if (mutation.meta?.silent) return;
            toast.error(errorMessage(err), err instanceof ApiError && err.requestId ? { description: `Request ID ${err.requestId}` } : undefined);
          },
        }),
      }),
  );
  return (
    <ThemeProvider>
      <QueryClientProvider client={client}>
        <AuthProvider>{children}</AuthProvider>
        <ThemedToaster />
      </QueryClientProvider>
    </ThemeProvider>
  );
}

function ThemedToaster() {
  const { resolved } = useTheme();
  return <Toaster theme={resolved} position="bottom-right" richColors closeButton toastOptions={{ className: "font-sans" }} />;
}
