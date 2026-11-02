"use client";

import { UploadCloud } from "lucide-react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useMemo, useState } from "react";

import { DocumentDrawer } from "@/components/knowledge/document-drawer";
import { DocumentsBrowser } from "@/components/knowledge/documents-browser";
import { Guarded, kbAbilities, usePerms } from "@/components/knowledge/shared";
import type { KnowledgeBase } from "@/components/knowledge/types";
import { UploadDialog } from "@/components/knowledge/upload-dialog";
import { Button } from "@/components/ui/button";
import { Page, PageHeader, Skeleton } from "@/components/ui/primitives";
import { useApi } from "@/lib/hooks";

export default function DocumentsPage() {
  return (
    <Suspense
      fallback={
        <Page>
          <Skeleton className="mb-8 h-24 w-full max-w-xl" />
          <Skeleton className="h-96 rounded-xl" />
        </Page>
      }
    >
      <DocumentLibrary />
    </Suspense>
  );
}

function DocumentLibrary() {
  const router = useRouter();
  const pathname = usePathname();
  const search = useSearchParams();
  const perms = usePerms();
  const docId = search.get("doc");
  const kbParam = search.get("kb_id") ?? undefined;
  const [uploadOpen, setUploadOpen] = useState(false);

  const kbs = useApi<KnowledgeBase[]>("/knowledge-bases");
  const canUpload = useMemo(() => (kbs.data ?? []).some((k) => kbAbilities(k, perms).upload), [kbs.data, perms]);

  const setDoc = useCallback(
    (id: string | null) => {
      const params = new URLSearchParams(search.toString());
      if (id) params.set("doc", id);
      else params.delete("doc");
      const qs = params.toString();
      router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
    },
    [router, pathname, search],
  );

  const uploadButton = (
    <Guarded allowed={canUpload || kbs.isLoading} reason="Uploading needs the kb.upload permission and write access to a knowledge base.">
      <Button onClick={() => setUploadOpen(true)} disabled={!canUpload}>
        <UploadCloud className="h-4 w-4" /> Upload
      </Button>
    </Guarded>
  );

  return (
    <Page>
      <PageHeader eyebrow="Knowledge" title="Documents" description="Every document you can access across knowledge bases — with processing status, parser output and index health." actions={uploadButton} />
      <DocumentsBrowser kbs={kbs.data} initialKbId={kbParam} onOpen={(id) => setDoc(id)} selectedId={docId} emptyAction={canUpload ? uploadButton : undefined} />
      <DocumentDrawer docId={docId} onClose={() => setDoc(null)} />
      <UploadDialog open={uploadOpen} onOpenChange={setUploadOpen} kbs={kbs.data} defaultKbId={kbParam} />
    </Page>
  );
}
