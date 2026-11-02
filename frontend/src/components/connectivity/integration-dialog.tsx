"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowUpRight, Blocks, BookOpen, Boxes, Cloud, FileText, FolderGit2, Globe, HardDrive, KeyRound, Library, Mail, Map as MapIcon, MessagesSquare, Rss } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Dialog, DialogClose, DialogContent } from "@/components/ui/dialog";
import { Field, Input, Select } from "@/components/ui/form";
import { Badge, Skeleton, Switch } from "@/components/ui/primitives";
import { patch, post } from "@/lib/api";
import { formatDate } from "@/lib/utils";

import { OAuthSetupNotice, useOAuthConnect } from "./oauth";
import { FormSection, Notice, ProviderStatusBadge, RadioCards, intervalOptions, invalidatePrefix, useKnowledgeBases, writableKbs } from "./shared";
import type { Integration, IntegrationAuthMode, IntegrationCatalogItem } from "./types";

export const INTEGRATION_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  website: Globe,
  sitemap: MapIcon,
  rss: Rss,
  github: FolderGit2,
  gitlab: FolderGit2,
  s3: HardDrive,
  google_drive: Cloud,
  onedrive: Cloud,
  dropbox: Boxes,
  sharepoint: FileText,
  notion: BookOpen,
  confluence: Library,
  slack: MessagesSquare,
  teams: MessagesSquare,
  email: Mail,
};
export const integrationIcon = (key: string) => INTEGRATION_ICONS[key] ?? Blocks;

export const INTEGRATION_CATEGORIES: Record<string, string> = {
  web: "Web",
  development: "Development",
  storage: "Storage",
  document_management: "Document management",
  collaboration: "Collaboration",
};

type FieldSpec = {
  key: string;
  label: string;
  kind?: "text" | "url" | "number" | "list" | "select";
  required?: boolean;
  /** Only shown (and validated) in these auth modes. */
  modes?: IntegrationAuthMode[];
  placeholder?: string;
  hint?: string;
  min?: number;
  max?: number;
  def?: number | string;
  options?: { value: string; label: string }[];
  wide?: boolean;
};

const maxItems = (label: string, def = 500): FieldSpec => ({ key: "max_items", label, kind: "number", def, min: 1, max: 5000 });
const lookback: FieldSpec = { key: "lookback_days", label: "Lookback (days)", kind: "number", def: 30, min: 1, max: 365, hint: "Messages older than this are not synced." };
const tenant: FieldSpec = { key: "tenant", label: "Directory (tenant)", placeholder: "common", hint: "Optional. A directory id or verified domain; empty uses the deployment default." };

/** Config fields per implemented provider, mirroring the backend connectors. */
const FIELDS: Record<string, FieldSpec[]> = {
  website: [
    { key: "start_url", label: "Start URL", kind: "url", required: true, placeholder: "https://docs.example.com/", hint: "The crawl stays on this host and respects robots.txt." },
    { key: "max_pages", label: "Max pages", kind: "number", def: 50, min: 1, max: 2000 },
    { key: "max_depth", label: "Max link depth", kind: "number", def: 2, min: 0, max: 6, hint: "0 fetches only the start URL." },
    { key: "path_prefix", label: "Path prefix", placeholder: "/docs/", hint: "Only follow links whose path starts with this." },
  ],
  sitemap: [
    { key: "sitemap_url", label: "Sitemap URL", kind: "url", required: true, placeholder: "https://example.com/sitemap.xml", hint: "Sitemap indexes are followed (up to 50 sitemaps)." },
    { key: "max_pages", label: "Max pages", kind: "number", def: 200, min: 1, max: 5000 },
    { key: "path_prefix", label: "Path prefix", placeholder: "/blog/", hint: "Only include URLs whose path starts with this." },
  ],
  rss: [
    { key: "feed_url", label: "Feed URL", kind: "url", required: true, placeholder: "https://example.com/feed.xml" },
    { key: "max_items", label: "Max items", kind: "number", def: 100, min: 1, max: 1000 },
  ],
  github: [
    { key: "branch", label: "Branch", placeholder: "default branch", hint: "Leave empty for the repository's default branch." },
    { key: "path_prefix", label: "Path prefix", placeholder: "docs/", hint: "Only files under this path." },
    { key: "extensions", label: "File extensions", kind: "list", placeholder: "md, mdx, rst, txt", hint: "Comma-separated. Empty uses docs + common source files." },
  ],
  gitlab: [
    { key: "project", label: "Project", required: true, placeholder: "group/project", hint: "Path with namespace (subgroups allowed) or the numeric project id." },
    { key: "branch", label: "Branch", placeholder: "default branch", hint: "Leave empty for the project's default branch." },
    { key: "path_prefix", label: "Path prefix", placeholder: "docs/", hint: "Only files under this path." },
    { key: "extensions", label: "File extensions", kind: "list", placeholder: "md, mdx, rst, txt", hint: "Comma-separated. Empty uses docs + common source files." },
    { key: "base_url", label: "GitLab URL", kind: "url", placeholder: "https://gitlab.com", hint: "For self-managed GitLab. Empty uses gitlab.com." },
    { key: "max_files", label: "Max files", kind: "number", def: 500, min: 1, max: 5000 },
  ],
  s3: [
    { key: "bucket", label: "Bucket", required: true, placeholder: "company-docs" },
    { key: "prefix", label: "Key prefix", placeholder: "handbook/", hint: "Only objects under this prefix." },
    { key: "endpoint_url", label: "Endpoint URL", kind: "url", placeholder: "https://s3.eu-central-1.amazonaws.com", hint: "For S3-compatible services (MinIO, R2, Wasabi). Empty for AWS." },
    { key: "region", label: "Region", placeholder: "us-east-1" },
    { key: "access_key_id", label: "Access key ID", placeholder: "AKIA…" },
  ],
  google_drive: [
    { key: "folder_id", label: "Folder ID", placeholder: "1AbCdEfGh…", hint: "Optional. The last part of the folder URL; subfolders are included. Empty syncs every file you can open, shared drives too.", wide: true },
    maxItems("Max files"),
  ],
  onedrive: [
    { key: "path", label: "Folder path", placeholder: "/Documents/Handbook", hint: "Optional. Subfolders are included. Empty syncs the whole drive." },
    { key: "drive_id", label: "Drive ID", placeholder: "your OneDrive", hint: "Optional. Defaults to the signed-in user's OneDrive." },
    tenant,
    maxItems("Max files"),
  ],
  sharepoint: [
    { key: "hostname", label: "SharePoint host", required: true, placeholder: "contoso.sharepoint.com" },
    { key: "site_path", label: "Site path", placeholder: "/sites/Handbook", hint: "Empty for the root site." },
    { key: "library", label: "Document library", placeholder: "Documents", hint: "Optional. Empty syncs every library in the site." },
    { key: "path", label: "Folder path", placeholder: "/Policies", hint: "Optional. Applied inside each library." },
    tenant,
    maxItems("Max files"),
  ],
  dropbox: [
    { key: "path", label: "Folder path", placeholder: "/Team/Handbook", hint: "Optional. Subfolders are included. Empty syncs the whole Dropbox." },
    maxItems("Max files"),
  ],
  notion: [maxItems("Max pages")],
  confluence: [
    { key: "space_keys", label: "Space keys", kind: "list", required: true, placeholder: "ENG, HR", hint: "Current pages in these spaces are synced." },
    { key: "site_url", label: "Site URL", kind: "url", required: true, modes: ["token"], placeholder: "https://acme.atlassian.net", hint: "Your Confluence Cloud address." },
    { key: "site_url", label: "Site URL", kind: "url", modes: ["oauth"], placeholder: "https://acme.atlassian.net", hint: "Optional. Picks this site when the Atlassian account can open several." },
    { key: "email", label: "Account email", required: true, modes: ["token"], placeholder: "you@acme.com", hint: "The Atlassian account the API token belongs to." },
    maxItems("Max pages"),
  ],
  slack: [
    { key: "channels", label: "Channels", kind: "list", placeholder: "#general, C0123ABCD", hint: "Optional. Empty syncs every channel the bot is a member of. Invite the bot to private channels first.", wide: true },
    lookback,
    maxItems("Max channel-days"),
  ],
  teams: [
    { key: "team_id", label: "Team ID", required: true, placeholder: "00000000-0000-0000-0000-000000000000", hint: "The team's group id: Teams → team → Get link to team → groupId.", wide: true },
    { key: "channel_ids", label: "Channel IDs", kind: "list", placeholder: "19:…@thread.tacv2", hint: "Optional. Empty syncs every channel you can read.", wide: true },
    lookback,
    tenant,
    maxItems("Max channel-days"),
  ],
  email: [
    { key: "host", label: "IMAP server", required: true, placeholder: "imap.example.com" },
    { key: "security", label: "Encryption", kind: "select", def: "ssl", options: [{ value: "ssl", label: "SSL/TLS (port 993)" }, { value: "starttls", label: "STARTTLS (port 143)" }], hint: "Unencrypted IMAP is not supported." },
    { key: "port", label: "Port", kind: "number", def: 993, min: 1, max: 65535 },
    { key: "username", label: "Username", required: true, placeholder: "you@example.com" },
    { key: "folder", label: "Folder", placeholder: "INBOX", hint: "Each message becomes one document." },
    lookback,
    maxItems("Max messages"),
  ],
};

const SECRET: Record<string, { label: string; hint: string; required?: boolean; modes?: IntegrationAuthMode[] }> = {
  github: { label: "Personal access token (optional)", hint: "Needed for private repositories and higher rate limits. Read-only contents scope is enough." },
  gitlab: { label: "Personal access token (optional)", hint: "Needed for private projects. read_api and read_repository scopes are enough." },
  s3: { label: "Secret access key", hint: "Pairs with the access key ID. Use credentials limited to read access on this bucket.", required: true },
  notion: { label: "Internal integration token", hint: "From notion.so/my-integrations (starts with ntn_). Share the pages to sync with the integration.", required: true, modes: ["token"] },
  confluence: { label: "API token", hint: "Create one at id.atlassian.com → Security → API tokens.", required: true, modes: ["token"] },
  slack: { label: "Bot token", hint: "xoxb-… from your Slack app's OAuth & Permissions page (channels:read, channels:history, users:read).", required: true, modes: ["token"] },
  email: { label: "Password", hint: "Use an app password where the provider offers one; Gmail and Microsoft 365 require one (or OAuth) for IMAP.", required: true },
};

/** Labels for the non-OAuth option of providers that support both. */
const TOKEN_MODE: Record<string, { label: string; description: string }> = {
  notion: { label: "Integration token", description: "Paste an internal integration token. Works without a platform OAuth app." },
  confluence: { label: "Email + API token", description: "Basic authentication with an Atlassian API token for one Confluence Cloud site." },
  slack: { label: "Bot token", description: "Paste the xoxb- bot token of a Slack app installed in your workspace." },
};

type Values = Record<string, string>;

const visible = (fields: FieldSpec[], mode: IntegrationAuthMode) => fields.filter((f) => !f.modes || f.modes.includes(mode));

function toValues(provider: string, config: Record<string, unknown>): Values {
  const v: Values = {};
  (FIELDS[provider] ?? []).forEach((f) => {
    if (f.key in v) return;
    const raw = config[f.key];
    v[f.key] = Array.isArray(raw) ? raw.join(", ") : raw === undefined || raw === null ? (f.def !== undefined ? String(f.def) : "") : String(raw);
  });
  if (provider === "github") {
    const [owner = "", name = ""] = String(config.repository ?? "").split("/");
    v.owner = owner;
    v.repo = name;
  }
  return v;
}

function toConfig(provider: string, v: Values, mode: IntegrationAuthMode): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  visible(FIELDS[provider] ?? [], mode).forEach((f) => {
    const raw = (v[f.key] ?? "").trim();
    if (!raw) return;
    if (f.kind === "number") out[f.key] = Number(raw);
    else if (f.kind === "list")
      out[f.key] = raw
        .split(/[,\s]+/)
        .map((s) => s.trim().replace(/^\./, ""))
        .filter(Boolean);
    else out[f.key] = raw;
  });
  if (provider === "github") out.repository = `${(v.owner ?? "").trim()}/${(v.repo ?? "").trim()}`;
  return out;
}

function validate(provider: string, v: Values, mode: IntegrationAuthMode): Record<string, string> {
  const errs: Record<string, string> = {};
  visible(FIELDS[provider] ?? [], mode).forEach((f) => {
    const raw = (v[f.key] ?? "").trim();
    if (f.required && !raw) errs[f.key] = `${f.label} is required.`;
    if (raw && f.kind === "url" && !/^https?:\/\/[^\s/$.?#].[^\s]*$/i.test(raw)) errs[f.key] = "Enter a full http(s) URL.";
    if (raw && f.kind === "number") {
      const n = Number(raw);
      if (!Number.isInteger(n) || (f.min !== undefined && n < f.min) || (f.max !== undefined && n > f.max)) errs[f.key] = `Whole number between ${f.min} and ${f.max}.`;
    }
  });
  if (provider === "github") {
    if (!/^[A-Za-z0-9_.-]+$/.test((v.owner ?? "").trim())) errs.owner = "Owner (user or organization) is required.";
    if (!/^[A-Za-z0-9_.-]+$/.test((v.repo ?? "").trim())) errs.repo = "Repository name is required.";
  }
  return errs;
}

/** Key-order-insensitive comparison of two config objects. */
function sameConfig(a: Record<string, unknown>, b: Record<string, unknown>): boolean {
  const norm = (o: Record<string, unknown>) => JSON.stringify(Object.keys(o).sort().map((k) => [k, o[k]]));
  return norm(a) === norm(b);
}

function defaultMode(meta: IntegrationCatalogItem | undefined, existing?: Integration | null): IntegrationAuthMode {
  if (existing?.auth_mode) return existing.auth_mode;
  const modes = meta?.auth_modes ?? ["token"];
  // Prefer OAuth when this deployment has the vendor's app; otherwise the token alternative.
  if (modes.includes("oauth") && modes.length > 1) return meta?.oauth?.configured ? "oauth" : (modes.find((m) => m !== "oauth") ?? "oauth");
  return modes[0];
}

export function IntegrationDialog({
  open,
  onOpenChange,
  provider,
  existing,
  catalog,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  provider: IntegrationCatalogItem | null;
  existing?: Integration | null;
  /** Creatable providers (shown as a source switcher when creating); when editing, used for provider metadata. */
  catalog?: IntegrationCatalogItem[];
}) {
  const qc = useQueryClient();
  const kbs = useKnowledgeBases();
  const writable = writableKbs(kbs.data);
  const [picked, setPicked] = useState<string>(provider?.key ?? "");
  useEffect(() => {
    if (open) setPicked(provider?.key ?? "");
  }, [open, provider]);
  const key = existing?.provider ?? picked;
  const meta = catalog?.find((c) => c.key === key) ?? (provider?.key === key ? provider : undefined);
  const [name, setName] = useState("");
  const [values, setValues] = useState<Values>({});
  const [secret, setSecret] = useState("");
  const [kbId, setKbId] = useState("");
  const [interval, setIntervalMin] = useState(0);
  const [enabled, setEnabled] = useState(true);
  const [mode, setMode] = useState<IntegrationAuthMode>("token");
  const [errors, setErrors] = useState<Record<string, string>>({});
  const connect = useOAuthConnect();

  useEffect(() => {
    if (!open) return;
    setName(existing?.name ?? "");
    setSecret("");
    setKbId(existing?.kb_id ?? "");
    setIntervalMin(existing?.sync_interval_minutes ?? 0);
    setEnabled(existing?.enabled ?? true);
  }, [open, existing]);
  useEffect(() => {
    if (!open) return;
    setValues(toValues(key, existing?.config ?? {}));
    setMode(defaultMode(meta, existing));
    setErrors({});
    // eslint-disable-next-line react-hooks/exhaustive-deps -- reset when the dialog opens or the source changes, not on catalog refetches
  }, [open, existing, key]);

  const modes = meta?.auth_modes ?? (existing?.auth_mode ? [existing.auth_mode] : ["token"]);
  const oauthApp = meta?.oauth;
  const isOAuth = mode === "oauth" && Boolean(oauthApp ?? existing?.oauth);
  const vendor = oauthApp?.vendor ?? existing?.oauth?.vendor ?? "the vendor";
  const setupMissing = isOAuth && oauthApp ? !oauthApp.configured : false;
  const secretBase = SECRET[key];
  const secretSpec = secretBase && (!secretBase.modes || secretBase.modes.includes(mode)) ? secretBase : undefined;
  const label = existing?.provider_label ?? meta?.label ?? key;
  const Icon = integrationIcon(key);
  const fields = visible(FIELDS[key] ?? [], mode);
  const switchedToToken = Boolean(existing && existing.auth_mode === "oauth" && mode !== "oauth");

  const save = useMutation({
    mutationFn: (body: Record<string, unknown>) => (existing ? patch<Integration>(`/integrations/${existing.id}`, body) : post<Integration>("/integrations", body)),
    onSuccess: (i) => {
      invalidatePrefix(qc, "/integrations", "/connections");
      if (!existing && isOAuth) {
        toast.message(`Integration “${i.name}” added`, { description: `Opening ${vendor} to approve access…` });
        connect.mutate(i.id, { onError: () => onOpenChange(false) });
        return;
      }
      toast.success(existing ? `Saved “${i.name}”` : `Integration “${i.name}” added`, { description: existing ? undefined : "Run Test to verify access, then Sync now." });
      onOpenChange(false);
    },
  });

  /** Body for PATCH (only what changed); null when the form is invalid. */
  const editBody = (): Record<string, unknown> | null => {
    if (!existing) return null;
    const errs = validate(key, values, mode);
    if (name.trim().length < 2) errs.name = "At least 2 characters.";
    if (secretSpec?.required && !secret && (switchedToToken || !existing.has_secret)) errs.secret = `${secretSpec.label} is required.`;
    setErrors(errs);
    if (Object.keys(errs).length) return null;
    const body: Record<string, unknown> = {};
    if (name.trim() !== existing.name) body.name = name.trim();
    // PATCH replaces the whole config: keep keys this form does not manage, drop cleared fields.
    const managed = new Set([...(FIELDS[key] ?? []).map((f) => f.key), "repository", ...(modes.length > 1 ? ["auth_mode"] : [])]);
    const config = toConfig(key, values, mode);
    if (modes.length > 1) config.auth_mode = mode;
    const nextConfig = { ...Object.fromEntries(Object.entries(existing.config).filter(([k]) => !managed.has(k))), ...config };
    if (!sameConfig(nextConfig, existing.config)) body.config = nextConfig;
    if (secret && !isOAuth) body.secret = secret;
    if (kbId !== (existing.kb_id ?? "")) body.kb_id = kbId || null;
    if (interval !== (existing.sync_interval_minutes ?? 0)) body.sync_interval_minutes = interval;
    if (enabled !== existing.enabled) body.enabled = enabled;
    return body;
  };

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (setupMissing) return;
    if (existing) {
      const body = editBody();
      if (body === null) return;
      if (!Object.keys(body).length) return onOpenChange(false);
      save.mutate(body);
      return;
    }
    const errs = validate(key, values, mode);
    if (name.trim().length < 2) errs.name = "At least 2 characters.";
    if (!kbId) errs.kb = "Choose where synchronised documents go.";
    if (secretSpec?.required && !secret) errs.secret = `${secretSpec.label} is required.`;
    setErrors(errs);
    if (Object.keys(errs).length) return;
    const config = toConfig(key, values, mode);
    if (modes.includes("oauth")) config.auth_mode = mode;
    save.mutate({ name: name.trim(), provider: key, config, secret: !isOAuth && secret ? secret : null, kb_id: kbId || null, sync_interval_minutes: interval || null });
  };

  /** Save pending edits (if any), then leave for the vendor's consent screen. */
  const reconnect = async () => {
    if (!existing) return;
    const body = editBody();
    if (body === null) return;
    if (Object.keys(body).length) {
      try {
        await save.mutateAsync(body);
      } catch {
        return;
      }
    }
    connect.mutate(existing.id);
  };

  const set = (k: string, v: string) =>
    setValues((p) => {
      const next = { ...p, [k]: v };
      // Keep the IMAP port in step with the encryption choice unless it was customised.
      if (key === "email" && k === "security" && ["993", "143", ""].includes(p.port ?? "")) next.port = v === "starttls" ? "143" : "993";
      return next;
    });
  const busy = save.isPending || connect.isPending;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        wide
        title={
          <span className="flex items-center gap-3">
            <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg border bg-surface-2 text-fg shadow-[inset_0_1px_0_hsl(var(--highlight))]">
              <Icon className="h-4 w-4" />
            </span>
            <span className="min-w-0 truncate">{existing ? `Edit ${existing.name}` : `Connect ${label}`}</span>
          </span>
        }
        description={
          <span className="mt-1 flex flex-wrap items-center gap-1.5">
            <ProviderStatusBadge status={existing?.provider_status ?? meta?.status} />
            {(existing?.provider_status ?? meta?.status) === "beta" ? "Implemented and tested against stubbed APIs; not yet verified against the live service." : "Content is fetched, versioned and indexed; removed items are deleted on the next sync."}
          </span>
        }
        footer={
          <>
            <DialogClose asChild>
              <Button variant="secondary">Cancel</Button>
            </DialogClose>
            <Button type="submit" form="integration-form" loading={busy} disabled={setupMissing || (!existing && !writable.length)}>
              {existing ? (
                "Save changes"
              ) : isOAuth ? (
                <>
                  Continue to {vendor} <ArrowUpRight className="h-3.5 w-3.5" />
                </>
              ) : (
                "Add integration"
              )}
            </Button>
          </>
        }
      >
        <form id="integration-form" onSubmit={submit} className="space-y-5" noValidate>
          <FormSection title="Source" description={!existing && catalog?.length ? "Switch the source type, then name this integration." : "How this integration appears in the list."}>
            <div className="grid gap-4 sm:grid-cols-2">
              {!existing && catalog?.length ? (
                <Field label="Source type" htmlFor="i-provider" hint={meta?.auth && meta.auth !== "none" ? `Auth: ${meta.auth}` : "No credentials needed."}>
                  <Select id="i-provider" value={picked} onChange={(e) => setPicked(e.target.value)}>
                    {catalog.map((c) => (
                      <option key={c.key} value={c.key}>
                        {c.label}
                        {c.status === "beta" ? " (beta)" : ""}
                      </option>
                    ))}
                  </Select>
                </Field>
              ) : null}
              <Field label="Name" htmlFor="i-name" error={errors.name} className={!existing && catalog?.length ? undefined : "sm:col-span-2"}>
                <Input id="i-name" value={name} onChange={(e) => setName(e.target.value)} placeholder={`e.g. ${label} — product docs`} maxLength={160} autoFocus />
              </Field>
            </div>
          </FormSection>

          {modes.includes("oauth") ? (
            <FormSection
              title="Connection"
              description={isOAuth ? `You approve read-only access on ${vendor}'s own page. Tokens are stored encrypted, refreshed automatically and never shown.` : "Credentials are encrypted at rest and never shown again."}
            >
              {modes.length > 1 ? (
                <RadioCards<IntegrationAuthMode>
                  name="Authentication"
                  layout="stack"
                  value={mode}
                  onChange={setMode}
                  options={[
                    { key: "oauth", label: `Connect with ${vendor}`, description: oauthApp && !oauthApp.configured ? "Needs a platform OAuth app on this deployment (see below)." : `Sign in at ${vendor} and approve access. Recommended.` },
                    { key: modes.find((m) => m !== "oauth") ?? "token", label: TOKEN_MODE[key]?.label ?? "Token", description: TOKEN_MODE[key]?.description ?? "Paste a token." },
                  ]}
                />
              ) : null}
              {isOAuth && oauthApp && setupMissing ? <OAuthSetupNotice app={oauthApp} label={label} /> : null}
              {isOAuth && !setupMissing && existing ? (
                <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border bg-surface-2/40 px-3.5 py-3">
                  <div className="min-w-0 text-[13px]">
                    {existing.oauth?.connected ? (
                      <>
                        <Badge tone="ok">
                          <KeyRound className="h-2.5 w-2.5" /> authorized
                        </Badge>{" "}
                        <span className="text-muted">
                          {existing.oauth.account ? <>as <span className="font-medium text-fg">{existing.oauth.account}</span></> : `with ${vendor}`}
                          {existing.oauth.connected_at ? <> · {formatDate(existing.oauth.connected_at)}</> : null}
                        </span>
                      </>
                    ) : (
                      <>
                        <Badge tone="warn">not authorized</Badge> <span className="text-muted">Approve access at {vendor} to test and sync.</span>
                      </>
                    )}
                  </div>
                  <Button type="button" size="sm" variant="secondary" loading={connect.isPending} disabled={save.isPending} onClick={reconnect}>
                    {existing.oauth?.connected ? `Reconnect with ${vendor}` : `Connect with ${vendor}`} <ArrowUpRight className="h-3 w-3" />
                  </Button>
                </div>
              ) : null}
              {isOAuth && !setupMissing && !existing ? (
                <p className="text-[12.5px] leading-relaxed text-muted">
                  After you continue, {vendor} asks you to approve {oauthApp?.scopes.length ? <>these scopes: <span className="font-mono text-[11.5px] text-fg">{oauthApp.scopes.join(" ")}</span></> : "access to the pages you choose"}. You come back here when it is done; the integration shows as connected only after a successful test.
                </p>
              ) : null}
              {switchedToToken ? <Notice tone="info">Switching to a token removes the stored {vendor} authorization when you save.</Notice> : null}
            </FormSection>
          ) : null}

          {key === "github" || fields.length || secretSpec ? (
            <FormSection title="Configuration" description={secretSpec ? "Secrets are encrypted at rest and never shown again." : "What to fetch from the source."}>
              <div className="grid gap-4 sm:grid-cols-2">
                {key === "github" ? (
                  <>
                    <Field label="Repository owner" htmlFor="i-owner" error={errors.owner}>
                      <Input id="i-owner" value={values.owner ?? ""} onChange={(e) => set("owner", e.target.value)} placeholder="acme" spellCheck={false} autoComplete="off" />
                    </Field>
                    <Field label="Repository name" htmlFor="i-repo" error={errors.repo}>
                      <Input id="i-repo" value={values.repo ?? ""} onChange={(e) => set("repo", e.target.value)} placeholder="handbook" spellCheck={false} autoComplete="off" />
                    </Field>
                  </>
                ) : null}
                {fields.map((f) => (
                  <Field key={`${f.key}-${f.modes?.join() ?? ""}`} label={f.label} htmlFor={`i-${f.key}`} error={errors[f.key]} hint={f.hint ?? (f.kind === "number" && f.min !== undefined ? `${f.min}–${f.max}` : undefined)} className={f.kind === "url" || f.wide ? "sm:col-span-2" : undefined}>
                    {f.kind === "select" ? (
                      <Select id={`i-${f.key}`} value={values[f.key] ?? String(f.def ?? "")} onChange={(e) => set(f.key, e.target.value)}>
                        {(f.options ?? []).map((o) => (
                          <option key={o.value} value={o.value}>
                            {o.label}
                          </option>
                        ))}
                      </Select>
                    ) : (
                      <Input
                        id={`i-${f.key}`}
                        value={values[f.key] ?? ""}
                        onChange={(e) => set(f.key, f.kind === "number" ? e.target.value.replace(/[^\d]/g, "") : e.target.value)}
                        placeholder={f.placeholder}
                        inputMode={f.kind === "number" ? "numeric" : f.kind === "url" ? "url" : undefined}
                        type={f.kind === "url" ? "url" : "text"}
                        spellCheck={false}
                        autoComplete="off"
                        className={f.kind === "url" || f.key.endsWith("_id") || f.key === "channel_ids" ? "font-mono text-[13px]" : undefined}
                      />
                    )}
                  </Field>
                ))}
                {secretSpec ? (
                  <Field
                    label={
                      <span className="flex items-center gap-2">
                        {secretSpec.label}
                        {existing?.has_secret && !switchedToToken ? <Badge tone="ok">stored</Badge> : null}
                      </span>
                    }
                    htmlFor="i-secret"
                    error={errors.secret}
                    hint={existing?.has_secret && !switchedToToken ? "A secret is stored (encrypted). Leave empty to keep it; enter a new value to replace it." : secretSpec.hint}
                    className="sm:col-span-2"
                  >
                    <Input id="i-secret" type="password" value={secret} onChange={(e) => setSecret(e.target.value)} autoComplete="new-password" placeholder={existing?.has_secret && !switchedToToken ? "••••••••  (unchanged)" : undefined} />
                  </Field>
                ) : null}
              </div>
            </FormSection>
          ) : null}

          <FormSection title="Destination" description="Where synchronised items are indexed, and how often.">
            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Target knowledge base" htmlFor="i-kb" error={errors.kb} hint="Synchronised items become documents here.">
                {kbs.isLoading ? (
                  <Skeleton className="h-9" />
                ) : writable.length || existing?.kb_id ? (
                  <Select id="i-kb" value={kbId} onChange={(e) => setKbId(e.target.value)}>
                    <option value="">{existing ? "None" : "Choose…"}</option>
                    {existing?.kb_id && !writable.some((k) => k.id === existing.kb_id) ? <option value={existing.kb_id}>Current (not writable by you)</option> : null}
                    {writable.map((k) => (
                      <option key={k.id} value={k.id}>
                        {k.name}
                      </option>
                    ))}
                  </Select>
                ) : (
                  <div className="rounded-lg border border-dashed px-3 py-2 text-xs text-muted">
                    No knowledge base you can write to.{" "}
                    <Link href="/knowledge-bases" className="font-medium text-accent hover:underline">
                      Create one first
                    </Link>
                    .
                  </div>
                )}
              </Field>
              <Field label="Sync schedule" htmlFor="i-int" hint="Scheduled syncs only fetch changed content.">
                <Select id="i-int" value={interval} onChange={(e) => setIntervalMin(Number(e.target.value))}>
                  {intervalOptions(existing?.sync_interval_minutes, 15).map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </Select>
              </Field>
            </div>
          </FormSection>

          {existing ? (
            <div className="flex items-center justify-between gap-4 rounded-xl border bg-surface-2/40 px-3.5 py-3">
              <div>
                <div className="text-[13px] font-medium">Enabled</div>
                <p className="mt-0.5 text-[12px] text-muted">Disabled integrations are skipped by the scheduler and by Sync now.</p>
              </div>
              <Switch checked={enabled} onCheckedChange={setEnabled} label="Integration enabled" />
            </div>
          ) : (
            <Notice tone="info">{isOAuth ? `The integration is saved first, then ${vendor} opens. It shows as connected only after authorization and a successful test.` : "The integration starts as “pending”. It only shows as connected after a successful Test or Sync."}</Notice>
          )}
        </form>
      </DialogContent>
    </Dialog>
  );
}
