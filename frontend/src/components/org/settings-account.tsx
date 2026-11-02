"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Laptop, LogOut, MailWarning, ShieldCheck, ShieldOff, Smartphone, Trash2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { Callout, CancelButton, CopyField, KV, SettingsSection, When } from "@/components/org/kit";
import type { Session } from "@/components/org/types";
import { Button } from "@/components/ui/button";
import { ConfirmDialog, Dialog, DialogContent } from "@/components/ui/dialog";
import { Field, Input, Select } from "@/components/ui/form";
import { Avatar, Badge, Card, CardBody, Mono, SkeletonRows } from "@/components/ui/primitives";
import { del, post, patch } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { cn, formatDate } from "@/lib/utils";

const LOCALES = ["en", "en-US", "en-GB", "de", "fr", "es", "it", "pt", "pt-BR", "nl", "sv", "da", "nb", "fi", "pl", "cs", "tr", "ar", "he", "ur", "hi", "ja", "ko", "zh-CN", "zh-TW"];

function localeName(code: string) {
  try {
    return new Intl.DisplayNames([code], { type: "language" }).of(code) || code;
  } catch {
    return code;
  }
}

/** Footer bar for a settings card: optional note on the left, actions on the right. */
function CardFooter({ children, note }: { children: React.ReactNode; note?: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-center justify-end gap-3 rounded-b-xl border-t bg-surface-2/30 px-5 py-3">
      {note ? <span className="mr-auto text-[12px] text-subtle">{note}</span> : null}
      {children}
    </div>
  );
}

/* ------------------------------------------------------------------ profile */
export function ProfileTab() {
  const { user, reload } = useAuth();
  const [name, setName] = useState("");
  const [locale, setLocale] = useState("en");
  useEffect(() => {
    if (user) {
      setName(user.full_name || "");
      setLocale(user.locale || "en");
    }
  }, [user]);
  const options = useMemo(() => (user?.locale && !LOCALES.includes(user.locale) ? [user.locale, ...LOCALES] : LOCALES), [user?.locale]);
  const m = useMutation({
    mutationFn: () => patch("/auth/me", { full_name: name.trim(), locale }),
    onSuccess: async () => {
      await reload();
      toast.success("Profile saved");
    },
  });
  const dirty = user && (name.trim() !== (user.full_name || "") || locale !== (user.locale || "en"));
  if (!user) return null;
  return (
    <div>
      <SettingsSection title="Your profile" description="How you appear to teammates across every organization you belong to.">
        <Card>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (dirty) m.mutate();
            }}
          >
            <div className="flex items-center gap-4 border-b px-5 py-4">
              <Avatar name={name || user.full_name} email={user.email} className="h-12 w-12 text-[15px]" />
              <div className="min-w-0">
                <div className="truncate text-[15px] font-semibold tracking-[-0.01em]">{name.trim() || user.email}</div>
                <div className="truncate text-[13px] text-muted">{user.email}</div>
              </div>
            </div>
            <CardBody className="space-y-4 py-5">
              <Field label="Full name" htmlFor="p-name">
                <Input id="p-name" value={name} onChange={(e) => setName(e.target.value)} maxLength={200} autoComplete="name" />
              </Field>
              <Field label="Email" htmlFor="p-email" hint="Your sign-in email cannot be changed here.">
                <Input id="p-email" value={user.email} disabled readOnly />
              </Field>
              <Field label="Language & region" htmlFor="p-locale" hint="Used for dates, numbers and notification emails.">
                <Select id="p-locale" value={locale} onChange={(e) => setLocale(e.target.value)}>
                  {options.map((l) => (
                    <option key={l} value={l}>
                      {localeName(l)} ({l})
                    </option>
                  ))}
                </Select>
              </Field>
            </CardBody>
            <CardFooter note={dirty ? "You have unsaved changes." : undefined}>
              <Button type="submit" size="sm" loading={m.isPending} disabled={!dirty || !name.trim()}>
                Save profile
              </Button>
            </CardFooter>
          </form>
        </Card>
      </SettingsSection>
      <SettingsSection title="Account" description="Verification status, sign-in history and identifiers for your user account.">
        <Card>
          <CardBody className="divide-y py-1.5">
            <KV label="Email verified">{user.email_verified_at ? <Badge tone="ok">verified {formatDate(user.email_verified_at, false)}</Badge> : <Badge tone="warn">not verified</Badge>}</KV>
            <KV label="Two-factor">{user.mfa_enabled ? <Badge tone="ok">enabled</Badge> : <Badge>off</Badge>}</KV>
            <KV label="Member since">{formatDate(user.created_at, false)}</KV>
            <KV label="Last sign-in">
              <When iso={user.last_login_at} />
            </KV>
            {user.is_platform_admin ? (
              <KV label="Platform role">
                <Badge tone="accent">platform administrator</Badge>
              </KV>
            ) : null}
            <KV label="User id">
              <Mono className="text-subtle">{user.id.slice(0, 8)}…</Mono>
            </KV>
          </CardBody>
        </Card>
      </SettingsSection>
    </div>
  );
}

/* ----------------------------------------------------------------- security */
function describeUA(ua: string) {
  const browser = /Edg\//.test(ua) ? "Edge" : /OPR\//.test(ua) ? "Opera" : /Firefox\//.test(ua) ? "Firefox" : /Chrome\//.test(ua) ? "Chrome" : /Safari\//.test(ua) ? "Safari" : /curl|python|httpx|node|axios|okhttp/i.test(ua) ? "API client" : "Unknown browser";
  const os = /Android/.test(ua) ? "Android" : /iPhone|iPad|iPod/.test(ua) ? "iOS" : /Windows/.test(ua) ? "Windows" : /Mac OS X|Macintosh/.test(ua) ? "macOS" : /CrOS/.test(ua) ? "ChromeOS" : /Linux/.test(ua) ? "Linux" : "";
  return { label: os ? `${browser} on ${os}` : browser, mobile: /Android|iPhone|iPad|iPod|Mobile/.test(ua) };
}

export function SecurityTab() {
  const { user } = useAuth();
  if (!user) return null;
  return (
    <div>
      {!user.email_verified_at ? (
        <div className="mb-8">
          <VerifyEmailCallout />
        </div>
      ) : null}
      <SettingsSection title="Password" description="Use a long, unique password. Changing it signs out every other session.">
        <PasswordCard />
      </SettingsSection>
      <SettingsSection
        title="Two-factor authentication"
        description="A time-based code from an authenticator app is required at every sign-in."
        aside={
          user.mfa_enabled ? (
            <Badge tone="ok">
              <ShieldCheck className="h-3 w-3" /> On
            </Badge>
          ) : (
            <Badge>
              <ShieldOff className="h-3 w-3" /> Off
            </Badge>
          )
        }
      >
        <MfaCard />
      </SettingsSection>
      <SettingsSection title="Active sessions" description="Devices currently signed in to your account. Revoke anything you don't recognise, then change your password.">
        <SessionsCard />
      </SettingsSection>
      <SettingsSection title={<span className="text-danger">Danger zone</span>} description="Signs you out everywhere and blocks sign-in. Organizations you own keep running; make sure another owner exists first.">
        <DeactivateCard />
      </SettingsSection>
    </div>
  );
}

function VerifyEmailCallout() {
  const resend = useMutation({
    mutationFn: () => post<{ detail?: string }>("/auth/resend-verification"),
    onSuccess: (r) => toast.success(r.detail || "Verification email sent"),
  });
  return (
    <Callout
      tone="warn"
      icon={<MailWarning className="h-4 w-4" />}
      title="Your email address is not verified"
      action={
        <Button size="sm" variant="secondary" loading={resend.isPending} onClick={() => resend.mutate()}>
          Resend link
        </Button>
      }
    >
      Verify it so password resets and security alerts reach you. Links expire, so use the most recent email.
    </Callout>
  );
}

function PasswordCard() {
  const qc = useQueryClient();
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const m = useMutation({
    mutationFn: () => post<{ detail?: string }>("/auth/password/change", { current_password: current, new_password: next }),
    onSuccess: (r) => {
      toast.success(r.detail || "Password changed");
      setCurrent("");
      setNext("");
      setConfirm("");
      qc.invalidateQueries({ queryKey: ["/auth/sessions"] });
    },
  });
  const classes = [/[a-z]/, /[A-Z]/, /\d/, /[^A-Za-z0-9]/].filter((r) => r.test(next)).length;
  const weak = next.length > 0 && (next.length < 10 || classes < 3);
  const mismatch = confirm.length > 0 && confirm !== next;
  const strength = next.length === 0 ? 0 : Math.min(4, (next.length >= 10 ? 1 : 0) + Math.max(0, classes - 1));
  return (
    <Card>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (current && next && !weak && !mismatch) m.mutate();
        }}
      >
        <CardBody className="space-y-4 py-5">
          <Field label="Current password" htmlFor="pw-current">
            <Input id="pw-current" type="password" autoComplete="current-password" value={current} onChange={(e) => setCurrent(e.target.value)} />
          </Field>
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="New password" htmlFor="pw-new" error={weak ? "At least 10 characters, mixing three of: lowercase, uppercase, digits, symbols." : null} hint="At least 10 characters, mixing three of: lowercase, uppercase, digits, symbols.">
              <Input id="pw-new" type="password" autoComplete="new-password" value={next} onChange={(e) => setNext(e.target.value)} />
            </Field>
            <Field label="Confirm new password" htmlFor="pw-confirm" error={mismatch ? "Passwords do not match." : null}>
              <Input id="pw-confirm" type="password" autoComplete="new-password" value={confirm} onChange={(e) => setConfirm(e.target.value)} />
            </Field>
          </div>
          {next ? (
            <div className="flex items-center gap-3" aria-hidden>
              <div className="flex flex-1 gap-1">
                {[1, 2, 3, 4].map((i) => (
                  <span key={i} className={cn("h-1 flex-1 rounded-full transition-colors", i <= strength ? (weak ? "bg-warn" : "bg-ok") : "bg-surface-2 ring-1 ring-inset ring-border/60")} />
                ))}
              </div>
              <span className={cn("w-20 text-right text-[11.5px] font-medium", weak ? "text-warn" : "text-ok")}>{weak ? "Too weak" : "Strong enough"}</span>
            </div>
          ) : null}
        </CardBody>
        <CardFooter>
          <Button type="submit" size="sm" loading={m.isPending} disabled={!current || !next || !confirm || weak || mismatch}>
            Change password
          </Button>
        </CardFooter>
      </form>
    </Card>
  );
}

function MfaCard() {
  const { user, reload } = useAuth();
  const [setup, setSetup] = useState<{ secret: string; otpauth_uri: string } | null>(null);
  const [code, setCode] = useState("");
  const [disabling, setDisabling] = useState(false);
  const begin = useMutation({
    mutationFn: () => post<{ secret: string; otpauth_uri: string }>("/auth/mfa/setup"),
    onSuccess: (r) => {
      setSetup(r);
      setCode("");
    },
  });
  const confirmM = useMutation({
    mutationFn: () => post<{ detail?: string }>("/auth/mfa/confirm", { code: code.replace(/\s/g, "") }),
    onSuccess: async (r) => {
      setSetup(null);
      setCode("");
      await reload();
      toast.success(r.detail || "Two-factor authentication enabled");
    },
  });
  const grouped = setup?.secret.replace(/(.{4})/g, "$1 ").trim() ?? "";

  return (
    <Card>
      {user?.mfa_enabled ? (
        <>
          <div className="flex items-start gap-3.5 px-5 py-5">
            <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl border border-ok/25 bg-ok/[0.07] text-ok">
              <ShieldCheck className="h-5 w-5" />
            </span>
            <div className="min-w-0">
              <div className="text-[14px] font-medium">Protected with an authenticator app</div>
              <p className="mt-0.5 text-[13px] leading-relaxed text-muted">To move to a new device, disable two-factor and set it up again.</p>
            </div>
          </div>
          <CardFooter>
            <Button size="sm" variant="secondary" onClick={() => setDisabling(true)}>
              Disable two-factor
            </Button>
          </CardFooter>
        </>
      ) : setup ? (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (code.replace(/\s/g, "").length >= 6) confirmM.mutate();
          }}
        >
          <CardBody className="space-y-4 py-5">
            <Callout tone="warn" title="Shown only for this setup">
              Add the key to an authenticator app (1Password, Google Authenticator, Authy, Microsoft Authenticator…) now. Choose “enter a setup key” — or paste the otpauth link into an app that accepts it. No QR code is generated so the secret never leaves this page as an image.
            </Callout>
            <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-end">
              <CopyField label="Setup key" value={setup.secret} />
              <p className="rounded-lg border border-dashed px-3 py-2 font-mono text-[12px] tracking-[0.14em] text-muted" aria-hidden>
                {grouped}
              </p>
            </div>
            <CopyField label="otpauth link" value={setup.otpauth_uri} multiline />
            <Field label="6-digit code from the app" htmlFor="mfa-code">
              <Input id="mfa-code" inputMode="numeric" autoComplete="one-time-code" maxLength={8} value={code} onChange={(e) => setCode(e.target.value.replace(/[^\d\s]/g, ""))} className="block h-11 max-w-[220px] font-mono text-[17px] tracking-[0.35em]" autoFocus />
            </Field>
          </CardBody>
          <CardFooter>
            <Button type="button" size="sm" variant="secondary" onClick={() => setSetup(null)}>
              Cancel
            </Button>
            <Button type="submit" size="sm" loading={confirmM.isPending} disabled={code.replace(/\s/g, "").length < 6}>
              Verify & enable
            </Button>
          </CardFooter>
        </form>
      ) : (
        <>
          <div className="flex items-start gap-3.5 px-5 py-5">
            <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl border bg-surface-2/60 text-subtle">
              <ShieldOff className="h-5 w-5" />
            </span>
            <div className="min-w-0">
              <div className="text-[14px] font-medium">Two-factor is off</div>
              <p className="mt-0.5 text-[13px] leading-relaxed text-muted">Protect your account against stolen passwords. Some organizations require two-factor for every member.</p>
            </div>
          </div>
          <CardFooter>
            <Button size="sm" onClick={() => begin.mutate()} loading={begin.isPending}>
              <ShieldCheck className="h-3.5 w-3.5" /> Set up two-factor
            </Button>
          </CardFooter>
        </>
      )}
      <DisableMfaDialog open={disabling} onOpenChange={setDisabling} />
    </Card>
  );
}

function DisableMfaDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (v: boolean) => void }) {
  const { reload } = useAuth();
  const [password, setPassword] = useState("");
  const [code, setCode] = useState("");
  useEffect(() => {
    if (!open) {
      setPassword("");
      setCode("");
    }
  }, [open]);
  const m = useMutation({
    mutationFn: () => post<{ detail?: string }>("/auth/mfa/disable", { password, code: code.replace(/\s/g, "") }),
    onSuccess: async (r) => {
      onOpenChange(false);
      await reload();
      toast.success(r.detail || "Two-factor authentication disabled");
    },
  });
  const valid = password && code.replace(/\s/g, "").length >= 6;
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        title="Disable two-factor authentication?"
        description="Your account will be protected by your password only. Confirm with your password and a current code."
        footer={
          <>
            <CancelButton />
            <Button variant="danger" type="submit" form="mfa-disable" loading={m.isPending} disabled={!valid}>
              Disable
            </Button>
          </>
        }
      >
        <form
          id="mfa-disable"
          className="space-y-4"
          onSubmit={(e) => {
            e.preventDefault();
            if (valid) m.mutate();
          }}
        >
          <Field label="Password" htmlFor="mfa-d-pw">
            <Input id="mfa-d-pw" type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} autoFocus />
          </Field>
          <Field label="Authentication code" htmlFor="mfa-d-code">
            <Input id="mfa-d-code" inputMode="numeric" autoComplete="one-time-code" maxLength={8} value={code} onChange={(e) => setCode(e.target.value.replace(/[^\d\s]/g, ""))} className="font-mono tracking-[0.3em]" />
          </Field>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function SessionsCard() {
  const qc = useQueryClient();
  const sessions = useApi<Session[]>("/auth/sessions");
  const [revoking, setRevoking] = useState<Session | null>(null);
  const [revokingOthers, setRevokingOthers] = useState(false);
  const others = (sessions.data ?? []).filter((s) => !s.current);
  const revoke = useMutation({
    mutationFn: (s: Session) => del(`/auth/sessions/${s.id}`),
    onSuccess: () => {
      toast.success("Session signed out");
      setRevoking(null);
      qc.invalidateQueries({ queryKey: ["/auth/sessions"] });
    },
  });
  const revokeAll = useMutation({
    mutationFn: async () => {
      for (const s of others) await del(`/auth/sessions/${s.id}`);
      return others.length;
    },
    onSuccess: (n) => {
      toast.success(`${n} other session${n === 1 ? "" : "s"} signed out`);
      setRevokingOthers(false);
      qc.invalidateQueries({ queryKey: ["/auth/sessions"] });
    },
    onSettled: () => qc.invalidateQueries({ queryKey: ["/auth/sessions"] }),
  });
  return (
    <Card className="overflow-hidden">
      <div className="flex flex-wrap items-center gap-3 border-b px-5 py-3">
        <span className="text-[13px] text-muted">
          {sessions.data ? (
            <>
              <span className="num font-mono font-medium text-fg">{sessions.data.length}</span> signed-in {sessions.data.length === 1 ? "device" : "devices"}
            </>
          ) : (
            "Loading sessions…"
          )}
        </span>
        {others.length ? (
          <Button size="xs" variant="secondary" className="ml-auto" onClick={() => setRevokingOthers(true)}>
            <LogOut className="h-3.5 w-3.5" /> Sign out other sessions
          </Button>
        ) : null}
      </div>
      {sessions.isLoading ? (
        <SkeletonRows rows={3} />
      ) : (
        <ul className="divide-y">
          {(sessions.data ?? []).map((s) => {
            const d = describeUA(s.user_agent || "");
            return (
              <li key={s.id} className="flex flex-wrap items-center gap-3.5 px-5 py-3.5">
                <span className={cn("grid h-9 w-9 shrink-0 place-items-center rounded-lg border", s.current ? "border-ok/25 bg-ok/[0.07] text-ok" : "bg-surface-2/60 text-muted")}>{d.mobile ? <Smartphone className="h-4 w-4" /> : <Laptop className="h-4 w-4" />}</span>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2 text-[13.5px] font-medium">
                    {d.label}
                    {s.current ? <Badge tone="ok">this device</Badge> : null}
                  </div>
                  <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[12px] text-subtle">
                    <Mono className="text-muted">{s.ip || "unknown IP"}</Mono>
                    <span>
                      signed in <When iso={s.created_at} />
                    </span>
                    <span>
                      active <When iso={s.last_used_at} />
                    </span>
                    <span title={formatDate(s.expires_at)}>expires {formatDate(s.expires_at, false)}</span>
                  </div>
                </div>
                {!s.current ? (
                  <Button size="xs" variant="ghost" onClick={() => setRevoking(s)} aria-label={`Sign out ${d.label}`}>
                    <Trash2 className="h-3.5 w-3.5" /> Revoke
                  </Button>
                ) : null}
              </li>
            );
          })}
        </ul>
      )}
      <ConfirmDialog
        open={Boolean(revoking)}
        onOpenChange={(v) => !v && setRevoking(null)}
        title="Sign out this session?"
        description={revoking ? `${describeUA(revoking.user_agent || "").label} (${revoking.ip || "unknown IP"}) will be signed out on its next request.` : undefined}
        confirmLabel="Sign out session"
        loading={revoke.isPending}
        onConfirm={() => {
          if (revoking) revoke.mutate(revoking);
        }}
      />
      <ConfirmDialog
        open={revokingOthers}
        onOpenChange={setRevokingOthers}
        title="Sign out all other sessions?"
        description={`${others.length} other session${others.length === 1 ? "" : "s"} will be signed out. This device stays signed in.`}
        confirmLabel="Sign out others"
        loading={revokeAll.isPending}
        onConfirm={() => revokeAll.mutate()}
      />
    </Card>
  );
}

function DeactivateCard() {
  const { user, signOut } = useAuth();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [password, setPassword] = useState("");
  const m = useMutation({
    mutationFn: () => post<{ detail?: string }>("/auth/deactivate", { password }),
    onSuccess: async (r) => {
      toast.success(r.detail || "Account deactivated");
      setOpen(false);
      await signOut();
      router.replace("/login");
    },
  });
  return (
    <Card className="border-danger/25">
      <div className="flex flex-wrap items-center gap-4 px-5 py-4">
        <div className="min-w-0 flex-1 basis-64">
          <div className="text-[14px] font-medium">Deactivate account</div>
          <p className="mt-0.5 text-[13px] leading-relaxed text-muted">Deactivation is immediate. A platform administrator can reactivate the account; your organization memberships are kept but unusable while deactivated.</p>
        </div>
        <Button variant="danger" size="sm" onClick={() => setOpen(true)}>
          Deactivate my account
        </Button>
      </div>
      <ConfirmDialog
        open={open}
        onOpenChange={(v) => {
          setOpen(v);
          if (!v) setPassword("");
        }}
        title="Deactivate your account?"
        description="You will be signed out of every device and won't be able to sign in again."
        confirmLabel="Deactivate account"
        requireText={user?.email}
        loading={m.isPending}
        onConfirm={() => {
          if (password) m.mutate();
          else toast.error("Enter your password to confirm.");
        }}
      >
        <Field label="Password" htmlFor="deact-pw" className="mb-4">
          <Input id="deact-pw" type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} />
        </Field>
      </ConfirmDialog>
    </Card>
  );
}
