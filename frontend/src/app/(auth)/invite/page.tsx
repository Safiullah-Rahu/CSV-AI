"use client";

import { Suspense } from "react";

import { AcceptInvite } from "../token-pages";

export default function Page() {
  return (
    <Suspense>
      <AcceptInvite />
    </Suspense>
  );
}
