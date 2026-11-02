"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Field, Input } from "@/components/ui/form";
import { errorMessage, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";

export default function RegisterPage() {
  const router = useRouter();
  const { signIn } = useAuth();
  const [form, setForm] = useState({ full_name: "", email: "", password: "", organization_name: "" });
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [verifyNotice, setVerifyNotice] = useState(false);
  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement>) => setForm((f) => ({ ...f, [k]: e.target.value }));

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res = await post<{ access_token: string | null }>("/auth/register", { ...form, organization_name: form.organization_name || null }, { noOrg: true });
      if (res.access_token) {
        await signIn(res.access_token);
        router.replace("/overview?welcome=1");
      } else {
        setVerifyNotice(true);
      }
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  if (verifyNotice) {
    return (
      <div>
        <h1 className="text-[26px] font-semibold leading-tight tracking-[-0.03em]">Check your inbox</h1>
        <p className="mt-2 text-[14px] leading-relaxed text-muted">We sent a verification link to {form.email}. Verify your address, then sign in.</p>
        <Link href="/login" className="mt-6 inline-block text-sm font-medium text-accent hover:underline">
          Back to sign in
        </Link>
      </div>
    );
  }

  return (
    <div>
      <h1 className="text-[26px] font-semibold leading-tight tracking-[-0.03em]">Create your account</h1>
      <p className="mt-2 text-[14px] leading-relaxed text-muted">Start a new organization, or join one later through an invitation.</p>
      <form className="mt-8 space-y-4" onSubmit={submit}>
        <Field htmlFor="reg-name" label="Full name">
          <Input id="reg-name" value={form.full_name} onChange={set("full_name")} autoComplete="name" autoFocus className="h-10" />
        </Field>
        <Field htmlFor="reg-email" label="Work email">
          <Input id="reg-email" type="email" value={form.email} onChange={set("email")} autoComplete="email" required className="h-10" />
        </Field>
        <Field htmlFor="reg-password" label="Password" hint="At least 10 characters, mixing three of: lowercase, uppercase, digits, symbols.">
          <Input id="reg-password" type="password" value={form.password} onChange={set("password")} autoComplete="new-password" required minLength={10} className="h-10" />
        </Field>
        <Field htmlFor="reg-org" label="Organization name" hint="Optional — you can create or join an organization afterwards.">
          <Input id="reg-org" value={form.organization_name} onChange={set("organization_name")} placeholder="Acme Corporation" className="h-10" />
        </Field>
        {error ? <p className="rounded-lg bg-danger/[0.07] px-3 py-2.5 text-[13px] text-danger ring-1 ring-inset ring-danger/20" role="alert">{error}</p> : null}
        <Button variant="ink" className="w-full" size="lg" loading={busy}>
          Create account
        </Button>
      </form>
      <p className="mt-8 border-t pt-6 text-center text-[13.5px] text-muted">
        Already have an account?{" "}
        <Link href="/login" className="font-medium text-accent hover:underline">
          Sign in
        </Link>
      </p>
    </div>
  );
}
