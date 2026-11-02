"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Field, Input } from "@/components/ui/form";
import { errorMessage, get, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";

function Notice({ tone, children }: { tone: "ok" | "danger"; children: React.ReactNode }) {
  return <p className={tone === "ok" ? "rounded-lg bg-ok/[0.07] px-3 py-2.5 text-[13px] text-ok ring-1 ring-inset ring-ok/20" : "rounded-lg bg-danger/[0.07] px-3 py-2.5 text-[13px] text-danger ring-1 ring-inset ring-danger/20"} role="status">{children}</p>;
}

export function ForgotPassword() {
  const [email, setEmail] = useState("");
  const [done, setDone] = useState(false);
  const [busy, setBusy] = useState(false);
  return (
    <div>
      <h1 className="text-[26px] font-semibold leading-tight tracking-[-0.03em]">Reset your password</h1>
      <p className="mt-2 text-[14px] leading-relaxed text-muted">We&apos;ll email you a link to choose a new password.</p>
      {done ? (
        <div className="mt-8 space-y-4">
          <Notice tone="ok">If an account exists for {email}, a reset link is on its way. The link expires in one hour.</Notice>
          <Link href="/login" className="text-sm font-medium text-accent hover:underline">Back to sign in</Link>
        </div>
      ) : (
        <form
          className="mt-8 space-y-4"
          onSubmit={async (e) => {
            e.preventDefault();
            setBusy(true);
            try {
              await post("/auth/password/forgot", { email }, { noOrg: true });
            } finally {
              setBusy(false);
              setDone(true);
            }
          }}
        >
          <Field htmlFor="forgot-email" label="Email">
            <Input id="forgot-email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoFocus className="h-10" />
          </Field>
          <Button variant="ink" className="w-full" size="lg" loading={busy}>Send reset link</Button>
        </form>
      )}
    </div>
  );
}

export function ResetPassword() {
  const token = useSearchParams().get("token") || "";
  const router = useRouter();
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  return (
    <div>
      <h1 className="text-[26px] font-semibold leading-tight tracking-[-0.03em]">Choose a new password</h1>
      <p className="mt-2 text-[14px] leading-relaxed text-muted">All other sessions will be signed out.</p>
      <form
        className="mt-8 space-y-4"
        onSubmit={async (e) => {
          e.preventDefault();
          setBusy(true);
          setError(null);
          try {
            await post("/auth/password/reset", { token, password }, { noOrg: true });
            router.replace("/login?reset=1");
          } catch (err) {
            setError(errorMessage(err));
          } finally {
            setBusy(false);
          }
        }}
      >
        <Field htmlFor="reset-password" label="New password">
          <Input id="reset-password" type="password" autoComplete="new-password" value={password} onChange={(e) => setPassword(e.target.value)} required autoFocus className="h-10" />
        </Field>
        {error ? <Notice tone="danger">{error}</Notice> : null}
        <Button variant="ink" className="w-full" size="lg" loading={busy} disabled={!token}>Update password</Button>
      </form>
    </div>
  );
}

export function VerifyEmail() {
  const token = useSearchParams().get("token") || "";
  const [state, setState] = useState<"working" | "ok" | "error">("working");
  const [message, setMessage] = useState("");
  useEffect(() => {
    post("/auth/verify-email", { token }, { noOrg: true })
      .then(() => setState("ok"))
      .catch((err) => {
        setState("error");
        setMessage(errorMessage(err));
      });
  }, [token]);
  return (
    <div>
      <h1 className="text-[26px] font-semibold leading-tight tracking-[-0.03em]">Email verification</h1>
      <div className="mt-8 space-y-4">
        {state === "working" ? <p className="text-sm text-muted">Verifying…</p> : state === "ok" ? <Notice tone="ok">Your email address is verified.</Notice> : <Notice tone="danger">{message}</Notice>}
        <Link href="/overview" className="text-sm font-medium text-accent hover:underline">Continue</Link>
      </div>
    </div>
  );
}

export function AcceptInvite() {
  const token = useSearchParams().get("token") || "";
  const router = useRouter();
  const { signIn } = useAuth();
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const lookup = useQuery({
    queryKey: ["invite", token],
    queryFn: () => get<{ organization_name: string; email: string; account_exists: boolean; expires_at: string }>("/invitations/lookup", { token }),
    enabled: Boolean(token),
    retry: false,
    meta: { silent: true },
  });
  if (lookup.isError) {
    return (
      <div>
        <h1 className="text-[26px] font-semibold leading-tight tracking-[-0.03em]">Invitation unavailable</h1>
        <div className="mt-6"><Notice tone="danger">{errorMessage(lookup.error)}</Notice></div>
      </div>
    );
  }
  const inv = lookup.data;
  return (
    <div>
      <h1 className="text-[26px] font-semibold leading-tight tracking-[-0.03em]">Join {inv?.organization_name ?? "…"}</h1>
      <p className="mt-2 text-[14px] leading-relaxed text-muted">
        {inv ? (inv.account_exists ? `Sign in as ${inv.email} to accept.` : `Create your account for ${inv.email}.`) : "Loading invitation…"}
      </p>
      {inv ? (
        <form
          className="mt-8 space-y-4"
          onSubmit={async (e) => {
            e.preventDefault();
            setBusy(true);
            setError(null);
            try {
              const res = await post<{ access_token: string }>("/invitations/accept", { token, password, full_name: fullName || null }, { noOrg: true });
              await signIn(res.access_token);
              router.replace("/overview?welcome=1");
            } catch (err) {
              setError(errorMessage(err));
            } finally {
              setBusy(false);
            }
          }}
        >
          {!inv.account_exists ? (
            <Field htmlFor="invite-name" label="Full name">
              <Input id="invite-name" value={fullName} onChange={(e) => setFullName(e.target.value)} autoFocus className="h-10" />
            </Field>
          ) : null}
          <Field htmlFor="invite-password" label={inv.account_exists ? "Password" : "Choose a password"} hint={inv.account_exists ? undefined : "At least 10 characters, mixing three character types."}>
            <Input id="invite-password" type="password" autoComplete={inv.account_exists ? "current-password" : "new-password"} value={password} onChange={(e) => setPassword(e.target.value)} required className="h-10" />
          </Field>
          {error ? <Notice tone="danger">{error}</Notice> : null}
          <Button variant="ink" className="w-full" size="lg" loading={busy}>Accept invitation</Button>
        </form>
      ) : null}
    </div>
  );
}
