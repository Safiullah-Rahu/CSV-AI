"use client";

import * as DialogPrimitive from "@radix-ui/react-dialog";
import { AlertTriangle, X } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/form";
import { cn } from "@/lib/utils";

export const Dialog = DialogPrimitive.Root;
export const DialogTrigger = DialogPrimitive.Trigger;

export function DialogContent({
  title,
  description,
  children,
  footer,
  className,
  wide,
}: {
  title: React.ReactNode;
  description?: React.ReactNode;
  children?: React.ReactNode;
  footer?: React.ReactNode;
  className?: string;
  wide?: boolean;
}) {
  return (
    <DialogPrimitive.Portal>
      <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-[hsl(var(--shadow)/0.32)] backdrop-blur-[3px] animate-fade-in dark:bg-black/60" />
      <DialogPrimitive.Content
        className={cn(
          "fixed left-1/2 top-1/2 z-50 flex max-h-[88vh] w-[calc(100vw-32px)] -translate-x-1/2 -translate-y-1/2 flex-col rounded-2xl border bg-surface shadow-float animate-scale-in focus:outline-none",
          wide ? "max-w-3xl" : "max-w-lg",
          className,
        )}
      >
        <div className="flex items-start justify-between gap-4 px-6 pb-2 pt-5">
          <div className="min-w-0">
            <DialogPrimitive.Title className="text-[16px] font-semibold tracking-[-0.02em]">{title}</DialogPrimitive.Title>
            {description ? <DialogPrimitive.Description className="mt-1 text-[13px] leading-relaxed text-muted">{description}</DialogPrimitive.Description> : <DialogPrimitive.Description className="sr-only">Dialog</DialogPrimitive.Description>}
          </div>
          <DialogPrimitive.Close className="-mr-1.5 rounded-lg p-1.5 text-subtle transition-colors hover:bg-surface-2 hover:text-fg" aria-label="Close">
            <X className="h-4 w-4" />
          </DialogPrimitive.Close>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-6 py-4">{children}</div>
        {footer ? <div className="flex items-center justify-end gap-2 rounded-b-2xl border-t bg-surface-2/40 px-6 py-3.5">{footer}</div> : null}
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  );
}

export const DialogClose = DialogPrimitive.Close;

/** Confirmation for destructive actions; optional type-to-confirm. */
export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel = "Confirm",
  danger = true,
  requireText,
  onConfirm,
  loading,
  children,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  title: string;
  description?: React.ReactNode;
  confirmLabel?: string;
  danger?: boolean;
  requireText?: string;
  onConfirm: () => void | Promise<void>;
  loading?: boolean;
  children?: React.ReactNode;
}) {
  const [typed, setTyped] = useState("");
  const blocked = Boolean(requireText) && typed !== requireText;
  return (
    <Dialog
      open={open}
      onOpenChange={(v) => {
        if (!v) setTyped("");
        onOpenChange(v);
      }}
    >
      <DialogContent
        title={
          <span className="flex items-center gap-2">
            {danger ? <AlertTriangle className="h-4 w-4 text-danger" /> : null}
            {title}
          </span>
        }
        description={description}
        footer={
          <>
            <DialogClose asChild>
              <Button variant="secondary">Cancel</Button>
            </DialogClose>
            <Button variant={danger ? "danger" : "primary"} disabled={blocked} loading={loading} onClick={() => onConfirm()}>
              {confirmLabel}
            </Button>
          </>
        }
      >
        {children}
        {requireText ? (
          <div className="space-y-2">
            <p className="text-[13px] text-muted">
              Type <span className="font-mono font-semibold text-fg">{requireText}</span> to confirm.
            </p>
            <Input value={typed} onChange={(e) => setTyped(e.target.value)} autoFocus aria-label="Confirmation text" />
          </div>
        ) : null}
      </DialogContent>
    </Dialog>
  );
}
