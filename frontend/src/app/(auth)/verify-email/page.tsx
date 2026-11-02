"use client";

import { Suspense } from "react";

import { VerifyEmail } from "../token-pages";

export default function Page() {
  return (
    <Suspense>
      <VerifyEmail />
    </Suspense>
  );
}
