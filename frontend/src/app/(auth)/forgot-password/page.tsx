"use client";

import { Suspense } from "react";

import { ForgotPassword } from "../token-pages";

export default function Page() {
  return (
    <Suspense>
      <ForgotPassword />
    </Suspense>
  );
}
