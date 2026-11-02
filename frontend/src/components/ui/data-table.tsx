"use client";

import { ArrowDown, ArrowUp, ChevronLeft, ChevronRight, Search } from "lucide-react";
import { useMemo, useState } from "react";

import { Input } from "@/components/ui/form";
import { EmptyState, SkeletonRows } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";

export type Column<T> = {
  key: string;
  header: React.ReactNode;
  cell: (row: T) => React.ReactNode;
  sortValue?: (row: T) => string | number | null | undefined;
  className?: string;
  headerClassName?: string;
};

/**
 * Searchable, sortable, paginated table. Client-side by default; pass
 * `total` + `page`/`onPageChange` for server-side pagination.
 */
export function DataTable<T>({
  rows,
  columns,
  rowKey,
  loading,
  searchable = true,
  searchPlaceholder = "Search…",
  filterText,
  toolbar,
  empty,
  pageSize = 25,
  onRowClick,
  selectedKey,
  dense,
}: {
  rows: T[] | undefined;
  columns: Column<T>[];
  rowKey: (row: T) => string;
  loading?: boolean;
  searchable?: boolean;
  searchPlaceholder?: string;
  filterText?: (row: T) => string;
  toolbar?: React.ReactNode;
  empty?: React.ReactNode;
  pageSize?: number;
  onRowClick?: (row: T) => void;
  selectedKey?: string | null;
  dense?: boolean;
}) {
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 } | null>(null);
  const [page, setPage] = useState(0);

  const filtered = useMemo(() => {
    let data = rows ?? [];
    if (q && filterText) {
      const needle = q.toLowerCase();
      data = data.filter((r) => filterText(r).toLowerCase().includes(needle));
    }
    if (sort) {
      const col = columns.find((c) => c.key === sort.key);
      if (col?.sortValue) {
        data = [...data].sort((a, b) => {
          const av = col.sortValue!(a);
          const bv = col.sortValue!(b);
          if (av === bv) return 0;
          if (av === null || av === undefined) return 1;
          if (bv === null || bv === undefined) return -1;
          return (av > bv ? 1 : -1) * sort.dir;
        });
      }
    }
    return data;
  }, [rows, q, sort, columns, filterText]);

  const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const current = Math.min(page, pages - 1);
  const visible = filtered.slice(current * pageSize, current * pageSize + pageSize);

  return (
    <div className="card overflow-hidden">
      {(searchable && filterText) || toolbar ? (
        <div className="flex flex-wrap items-center gap-2 border-b px-3.5 py-3">
          {searchable && filterText ? (
            <div className="relative w-full max-w-xs">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-subtle" />
              <Input
                value={q}
                onChange={(e) => {
                  setQ(e.target.value);
                  setPage(0);
                }}
                placeholder={searchPlaceholder}
                className="h-8 bg-surface-2/50 pl-8 text-[13px] shadow-none focus:bg-surface"
                aria-label="Search table"
              />
            </div>
          ) : null}
          <div className="ml-auto flex flex-wrap items-center gap-2">{toolbar}</div>
        </div>
      ) : null}
      {loading ? (
        <SkeletonRows rows={6} />
      ) : visible.length === 0 ? (
        empty ?? <EmptyState title={q ? "No matches" : "Nothing here yet"} description={q ? "Try a different search." : undefined} />
      ) : (
        <div className="relative overflow-x-auto">
          <table className="w-full text-[13px]">
            <thead>
              <tr className="border-b bg-surface-2/50">
                {columns.map((c) => (
                  <th key={c.key} className={cn("h-9 whitespace-nowrap px-4 text-left font-mono text-[10.5px] font-medium uppercase tracking-[0.1em] text-subtle", c.headerClassName)}>
                    {c.sortValue ? (
                      <button
                        className="inline-flex items-center gap-1 uppercase transition-colors hover:text-fg"
                        onClick={() => setSort((s) => (s?.key === c.key ? (s.dir === 1 ? { key: c.key, dir: -1 } : null) : { key: c.key, dir: 1 }))}
                      >
                        {c.header}
                        {sort?.key === c.key ? sort.dir === 1 ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" /> : null}
                      </button>
                    ) : (
                      c.header
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {visible.map((row) => {
                const key = rowKey(row);
                return (
                  <tr
                    key={key}
                    onClick={onRowClick ? () => onRowClick(row) : undefined}
                    className={cn("group/row border-b transition-colors last:border-0 hover:bg-surface-2/50", onRowClick && "cursor-pointer", selectedKey === key && "bg-accent/[0.06] shadow-[inset_2px_0_0_hsl(var(--accent))]")}
                  >
                    {columns.map((c) => (
                      <td key={c.key} className={cn("px-4 align-middle", dense ? "py-2" : "py-3", c.className)}>
                        {c.cell(row)}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {!loading && filtered.length > pageSize ? (
        <div className="flex items-center justify-between border-t bg-surface-2/30 px-4 py-2.5 text-[12px] text-muted">
          <span className="num">
            {current * pageSize + 1}–{Math.min(filtered.length, (current + 1) * pageSize)} of {filtered.length}
          </span>
          <div className="flex items-center gap-1">
            <button className="rounded p-1 hover:bg-surface-2 disabled:opacity-40" disabled={current === 0} onClick={() => setPage(current - 1)} aria-label="Previous page">
              <ChevronLeft className="h-4 w-4" />
            </button>
            <span className="num px-1">
              {current + 1} / {pages}
            </span>
            <button className="rounded p-1 hover:bg-surface-2 disabled:opacity-40" disabled={current >= pages - 1} onClick={() => setPage(current + 1)} aria-label="Next page">
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
