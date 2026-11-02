"use client";

import { Suspense } from "react";

import { ResetPassword } from "../token-pages";

export default function Page() {
  return (
    <Suspense>
      <ResetPassword />
    </Suspense>
  );
}
