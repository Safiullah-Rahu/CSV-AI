"use client";

import { BookOpen, ExternalLink, FileJson, KeyRound, Webhook } from "lucide-react";
import { Suspense, useEffect, useMemo, useState } from "react";

import { ApiKeysTab } from "@/components/org/dev-api-keys";
import { WebhooksTab } from "@/components/org/dev-webhooks";
import { Callout, CopyButton, CopyField, TabBar, useUrlTab } from "@/components/org/kit";
import { Card, CardBody, CardHeader, Mono, Page, PageHeader, Skeleton, Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/primitives";
import { useAuth } from "@/lib/auth";

const TABS = ["keys", "webhooks", "reference"] as const;

export default function DevelopersPage() {
  return (
    <Suspense fallback={<Page><Skeleton className="h-96 rounded-xl" /></Page>}>
      <DevelopersInner />
    </Suspense>
  );
}

function useBaseUrl() {
  const [origin, setOrigin] = useState("https://your-host");
  useEffect(() => setOrigin(window.location.origin), []);
  return `${origin}/api/v1`;
}

function DevelopersInner() {
  const { can } = useAuth();
  const available = useMemo(() => TABS.filter((t) => (t === "keys" ? can("admin.api_keys") : t === "webhooks" ? can("admin.platform_settings") : true)), [can]);
  const [tab, setTab] = useUrlTab(TABS, available[0] ?? "reference");
  const baseUrl = useBaseUrl();

  return (
    <Page>
      <PageHeader eyebrow="Organization" title="API & developers" description="Programmatic access to your organization's knowledge: scoped API keys, signed webhooks and the REST reference." />
      <Tabs value={tab} onValueChange={setTab}>
        <TabBar>
          <TabsList variant="line">
            {can("admin.api_keys") ? (
              <TabsTrigger value="keys">
                <KeyRound className="h-3.5 w-3.5" /> API keys
              </TabsTrigger>
            ) : null}
            {can("admin.platform_settings") ? (
              <TabsTrigger value="webhooks">
                <Webhook className="h-3.5 w-3.5" /> Webhooks
              </TabsTrigger>
            ) : null}
            <TabsTrigger value="reference">
              <BookOpen className="h-3.5 w-3.5" /> API reference
            </TabsTrigger>
          </TabsList>
        </TabBar>
        <TabsContent value="keys" className="animate-fade-in focus-visible:outline-none">
          <ApiKeysTab baseUrl={baseUrl} />
        </TabsContent>
        <TabsContent value="webhooks" className="animate-fade-in focus-visible:outline-none">
          <WebhooksTab />
        </TabsContent>
        <TabsContent value="reference" className="animate-fade-in focus-visible:outline-none">
          <ApiReference baseUrl={baseUrl} />
        </TabsContent>
      </Tabs>
    </Page>
  );
}

function CodeBlock({ code, label, lang = "bash" }: { code: string; label: string; lang?: string }) {
  return (
    <div className="overflow-hidden rounded-xl border bg-surface-2/40">
      <div className="flex items-center gap-2 border-b bg-surface-2/60 py-1 pl-3.5 pr-1.5">
        <span className="flex gap-1" aria-hidden>
          <span className="h-2 w-2 rounded-full bg-border-strong" />
          <span className="h-2 w-2 rounded-full bg-border-strong" />
          <span className="h-2 w-2 rounded-full bg-border-strong" />
        </span>
        <span className="label ml-2">{lang}</span>
        <CopyButton value={code} label={`Copy ${label}`} iconOnly variant="ghost" className="ml-auto" />
      </div>
      <pre className="overflow-x-auto px-4 py-3.5 font-mono text-[12px] leading-[1.75] text-fg" aria-label={label}>
        {code}
      </pre>
    </div>
  );
}

function Endpoint({ method, path }: { method: "GET" | "POST"; path: string }) {
  return (
    <span className="inline-flex min-w-0 items-center gap-2">
      <span className={method === "POST" ? "rounded-[5px] bg-accent/10 px-1.5 py-0.5 font-mono text-[10.5px] font-semibold tracking-[0.06em] text-accent" : "rounded-[5px] bg-surface-2 px-1.5 py-0.5 font-mono text-[10.5px] font-semibold tracking-[0.06em] text-muted ring-1 ring-inset ring-border"}>{method}</span>
      <span className="truncate font-mono text-[12.5px] font-normal text-fg">{path}</span>
    </span>
  );
}

const STATUS_CODES: [string, string][] = [
  ["401", "missing, revoked or expired key"],
  ["403", "scope or permission missing"],
  ["402", "plan limit would be exceeded"],
  ["422", "validation failed"],
  ["429", "rate limit — back off and retry"],
  ["501", "capability not configured"],
];

function ApiReference({ baseUrl }: { baseUrl: string }) {
  const ask = `curl -s -X POST "${baseUrl}/chat/ask" \\
  -H "Authorization: Bearer rk_xxxxxxxxxxxx.your-secret" \\
  -H "Content-Type: application/json" \\
  -d '{
    "question": "What is our parental leave policy?",
    "kb_ids": ["<knowledge-base-id>"],
    "policy": "strict"
  }'`;
  const kbs = `curl -s "${baseUrl}/knowledge-bases" \\
  -H "Authorization: Bearer rk_xxxxxxxxxxxx.your-secret"`;
  const errorShape = `{
  "error": {
    "code": "permission_denied",
    "message": "Missing permission: kb.chat",
    "details": null,
    "request_id": "…"
  }
}`;
  return (
    <div className="grid gap-6 lg:grid-cols-3">
      <div className="min-w-0 space-y-6 lg:col-span-2">
        <Card className="overflow-hidden">
          <CardHeader title={<Endpoint method="POST" path="/chat/ask" />} description={<>Non-streaming RAG answer with citations. Needs scopes <Mono>kb.view</Mono>, <Mono>kb.chat</Mono> and <Mono>conv.create</Mono>.</>} />
          <CardBody className="space-y-3.5">
            <CodeBlock code={ask} label="chat ask example" />
            <p className="text-[13px] leading-relaxed text-muted">
              Omit <Mono>kb_ids</Mono> to search every knowledge base the key's owner can read. Pass <Mono>conversation_id</Mono> from a previous response to continue a thread. The response includes <Mono>content</Mono>, <Mono>citations</Mono>, <Mono>confidence</Mono>, <Mono>grounded</Mono>, <Mono>usage</Mono> and <Mono>timings</Mono>.
            </p>
          </CardBody>
        </Card>
        <Card className="overflow-hidden">
          <CardHeader title={<Endpoint method="GET" path="/knowledge-bases" />} description={<>Returns only knowledge bases the key's owner can read. Needs scope <Mono>kb.view</Mono>.</>} />
          <CardBody>
            <CodeBlock code={kbs} label="knowledge bases example" />
          </CardBody>
        </Card>
        <Card className="overflow-hidden">
          <CardHeader title="Errors" description="Every error uses the same envelope. Quote the request id when contacting support." />
          <CardBody className="space-y-4">
            <CodeBlock code={errorShape} label="error envelope" lang="json" />
            <ul className="grid overflow-hidden rounded-xl border text-[13px] sm:grid-cols-2">
              {STATUS_CODES.map(([code, text], i) => (
                <li key={code} className={`flex items-baseline gap-3 px-3.5 py-2.5 ${i > 0 ? "border-t" : ""} ${i === 1 ? "sm:border-t-0" : ""} sm:[&:nth-child(even)]:border-l`}>
                  <Mono className="w-8 shrink-0 font-medium text-fg">{code}</Mono>
                  <span className="text-muted">{text}</span>
                </li>
              ))}
            </ul>
          </CardBody>
        </Card>
      </div>
      <div className="min-w-0 space-y-6">
        <Card>
          <CardHeader title="Base URL & authentication" />
          <CardBody className="space-y-4">
            <CopyField label="Base URL" value={baseUrl} />
            <div className="space-y-2 text-[13px] leading-relaxed text-muted">
              <p>
                Send <Mono className="text-fg">Authorization: Bearer rk_…</Mono> on every request. Keys are bound to one organization, so no <Mono>X-Org-Id</Mono> header is needed.
              </p>
              <p>A key can only use the scopes it was created with, and never more than its creator currently holds — removing a role from the creator narrows the key too.</p>
            </div>
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="OpenAPI schema" description="Machine-readable description of every endpoint" />
          <CardBody className="space-y-3">
            <a href="/api/v1/openapi.json" target="_blank" rel="noopener noreferrer" className="group flex items-center gap-3 rounded-xl border bg-surface px-3.5 py-3 text-[13px] font-medium shadow-xs transition-colors hover:border-border-strong hover:bg-surface-2/50">
              <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-accent/10 text-accent">
                <FileJson className="h-4 w-4" />
              </span>
              <span className="min-w-0 flex-1 truncate font-mono text-[12.5px]">/api/v1/openapi.json</span>
              <ExternalLink className="h-3.5 w-3.5 text-subtle transition-colors group-hover:text-fg" />
            </a>
            <Callout tone="neutral">
              Import it into Postman, Insomnia or Swagger Editor, or generate a typed client. The interactive Swagger UI (<Mono>/docs</Mono>) and ReDoc (<Mono>/redoc</Mono>) are served by the API server at its own origin; this web app only proxies <Mono>/api/*</Mono>, so they are reachable only where your deployment exposes the API host directly.
            </Callout>
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
