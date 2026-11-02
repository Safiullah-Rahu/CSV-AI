"use client";

import { Library } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Field, Select } from "@/components/ui/form";
import { EmptyState } from "@/components/ui/primitives";
import { useApi } from "@/lib/hooks";

import { kbAbilities, usePerms } from "./shared";
import type { KBOptions, KnowledgeBase } from "./types";
import { UploadZone } from "./upload-zone";

/** Pick a knowledge base you can write to, then drop files or folders into it. */
export function UploadDialog({ open, onOpenChange, kbs, defaultKbId }: { open: boolean; onOpenChange: (v: boolean) => void; kbs: KnowledgeBase[] | undefined; defaultKbId?: string }) {
  const perms = usePerms();
  const writable = (kbs ?? []).filter((k) => kbAbilities(k, perms).upload);
  const [kbId, setKbId] = useState("");
  const options = useApi<KBOptions>(open ? "/knowledge-bases/options" : null, undefined, { staleTime: 5 * 60_000 });

  useEffect(() => {
    if (!open) return;
    const preferred = writable.find((k) => k.id === defaultKbId) ?? writable[0];
    if (preferred && !writable.some((k) => k.id === kbId)) setKbId(preferred.id);
  }, [open, defaultKbId, writable, kbId]);

  const kb = writable.find((k) => k.id === kbId);
  const ab = kbAbilities(kb, perms);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent wide title="Upload documents" description="Files are stored, parsed, chunked and embedded into the knowledge base you choose.">
        {!writable.length ? (
          <EmptyState
            icon={<Library className="h-5 w-5" />}
            title="No knowledge base you can upload to"
            description="You need write access and the kb.upload permission on at least one knowledge base."
            action={
              <Link href="/knowledge-bases" className="text-[13px] font-medium text-accent hover:underline">
                Go to knowledge bases
              </Link>
            }
          />
        ) : (
          <div className="space-y-4">
            <Field label="Knowledge base" htmlFor="upload-kb">
              <Select id="upload-kb" value={kbId} onChange={(e) => setKbId(e.target.value)}>
                {writable.map((k) => (
                  <option key={k.id} value={k.id}>
                    {k.name}
                  </option>
                ))}
              </Select>
            </Field>
            {kb ? <UploadZone key={kb.id} kbId={kb.id} kbName={kb.name} options={options.data} allowed={ab.upload} canRestrict={ab.restrictDocuments} /> : null}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
