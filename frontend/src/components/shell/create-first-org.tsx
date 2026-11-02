"use client";

import { useMutation } from "@tanstack/react-query";
import { Building2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { Logo } from "@/components/shell/sidebar";
import { Button } from "@/components/ui/button";
import { Field, Input } from "@/components/ui/form";
import { post } from "@/lib/api";
import { useAuth } from "@/lib/auth";

export function CreateFirstOrg() {
  const { reload, switchOrg, signOut, user } = useAuth();
  const router = useRouter();
  const [name, setName] = useState("");
  const m = useMutation({
    mutationFn: () => post<{ id: string }>("/orgs", { name }, { noOrg: true }),
    onSuccess: async (org) => {
      await reload();
      switchOrg(org.id);
      router.push("/overview?welcome=1");
    },
  });
  return (
    <div className="relative grid min-h-screen place-items-center bg-frame px-5">
      <div className="pointer-events-none absolute inset-0 bg-dots opacity-60 [mask-image:radial-gradient(ellipse_at_center,black,transparent_65%)]" />
      <div className="relative w-full max-w-[440px] animate-fade-up">
        <div className="mb-10 flex justify-center">
          <Logo />
        </div>
        <div className="card overflow-hidden">
          <div className="px-7 pb-6 pt-7">
            <div className="label mb-3 flex items-center gap-2">
              <span className="h-1 w-3 rounded-full bg-accent" /> One last step
            </div>
            <h1 className="font-display text-[34px] leading-[1.05] tracking-[-0.015em]">Create your organization</h1>
            <p className="mt-2.5 text-[13.5px] leading-relaxed text-muted">
              Signed in as <span className="font-medium text-fg">{user?.email}</span>. Each organization is a fully isolated tenant with its own members, knowledge and settings. You can also join one through an invitation link.
            </p>
            <form
              className="mt-6 space-y-4"
              onSubmit={(e) => {
                e.preventDefault();
                m.mutate();
              }}
            >
              <Field htmlFor="first-org-name" label="Organization name">
                <Input id="first-org-name" value={name} onChange={(e) => setName(e.target.value)} placeholder="Acme Corporation" autoFocus className="h-10" />
              </Field>
              <Button variant="ink" size="lg" className="w-full" loading={m.isPending} disabled={name.trim().length < 2}>
                <Building2 /> Create organization
              </Button>
            </form>
          </div>
        </div>
        <button className="mx-auto mt-5 block text-[12.5px] text-subtle transition-colors hover:text-fg" onClick={() => signOut().then(() => router.replace("/login"))}>
          Sign out
        </button>
      </div>
    </div>
  );
}
