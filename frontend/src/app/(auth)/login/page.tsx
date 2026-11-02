"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Field, Input } from "@/components/ui/form";
import { errorMessage, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";

type TokenOut = { access_token: string | null; mfa_required: boolean; mfa_token: string | null };

function LoginForm() {
  const router = useRouter();
  const params = useSearchParams();
  const { signIn, status } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [mfaToken, setMfaToken] = useState<string | null>(null);
  const [code, setCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const next = params.get("next") || "/overview";

  useEffect(() => {
    if (status === "authenticated") router.replace(next);
  }, [status, next, router]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res = mfaToken
        ? await post<TokenOut>("/auth/mfa/verify", { mfa_token: mfaToken, code }, { noOrg: true })
        : await post<TokenOut>("/auth/login", { email, password }, { noOrg: true });
      if (res.mfa_required && res.mfa_token) {
        setMfaToken(res.mfa_token);
      } else if (res.access_token) {
        await signIn(res.access_token);
        router.replace(next);
      }
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <h1 className="text-[26px] font-semibold leading-tight tracking-[-0.03em]">{mfaToken ? "Two-factor authentication" : "Welcome back"}</h1>
      <p className="mt-2 text-[14px] leading-relaxed text-muted">{mfaToken ? "Enter the 6-digit code from your authenticator app." : "Sign in to your organization's workspace."}</p>
      <form className="mt-8 space-y-4" onSubmit={submit}>
        {mfaToken ? (
          <Field htmlFor="mfa-code" label="Authentication code">
            <Input id="mfa-code" inputMode="numeric" autoComplete="one-time-code" maxLength={8} value={code} onChange={(e) => setCode(e.target.value)} autoFocus className="h-12 text-center font-mono text-[18px] tracking-[0.4em]" />
          </Field>
        ) : (
          <>
            <Field label="Email" htmlFor="email">
              <Input id="email" type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoFocus className="h-10" />
            </Field>
            <Field
              label={
                <span className="flex w-full items-center justify-between">
                  Password
                  <Link href="/forgot-password" className="text-xs font-normal text-accent hover:underline">
                    Forgot?
                  </Link>
                </span>
              }
              htmlFor="password"
            >
              <Input id="password" type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} required className="h-10" />
            </Field>
          </>
        )}
        {error ? <p className="rounded-lg bg-danger/[0.07] px-3 py-2.5 text-[13px] text-danger ring-1 ring-inset ring-danger/20" role="alert">{error}</p> : null}
        <Button variant="ink" className="w-full" size="lg" loading={busy}>
          {mfaToken ? "Verify" : "Sign in"}
        </Button>
      </form>
      {!mfaToken ? (
        <p className="mt-8 border-t pt-6 text-center text-[13.5px] text-muted">
          New here?{" "}
          <Link href="/register" className="font-medium text-accent hover:underline">
            Create an account
          </Link>
        </p>
      ) : (
        <button className="mt-6 w-full text-center text-sm text-muted hover:text-fg" onClick={() => setMfaToken(null)}>
          Use a different account
        </button>
      )}
    </div>
  );
}

export default function LoginPage() {
  return (
    <Suspense>
      <LoginForm />
    </Suspense>
  );
}
