"use client";

/**
 * A small modal dialog on the native <dialog> element: the browser traps focus and keeps the page inert; Esc and a
 * click on the backdrop close it; focus returns to whatever opened it (or to the main region if that is gone).
 * Mark the element to focus first with `data-autofocus`.
 */

import { useEffect, useId, useRef, type ReactNode } from "react";

import { Icon } from "./Icon";

interface DialogProps {
  title: string;
  description?: ReactNode;
  /** Called on Esc, backdrop click or the close button. Ignored while `busy`. */
  onClose: () => void;
  busy?: boolean;
  role?: "dialog" | "alertdialog";
  children: ReactNode;
  size?: "sm" | "md";
}

export function restoreFocus(el: Element | null) {
  if (el instanceof HTMLElement && el.isConnected && el !== document.body) {
    el.focus({ preventScroll: true });
    if (document.activeElement === el) return;
  }
  document.getElementById("main")?.focus({ preventScroll: true });
}

export function Dialog({ title, description, onClose, busy = false, role = "dialog", children, size = "sm" }: DialogProps) {
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  const descId = useId();
  const pressedBackdrop = useRef(false);
  const onCloseRef = useRef(onClose);
  const busyRef = useRef(busy);
  useEffect(() => {
    onCloseRef.current = onClose;
    busyRef.current = busy;
  });

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    const returnTo = document.activeElement;
    if (!dialog.open) dialog.showModal();
    const first = dialog.querySelector<HTMLElement>("[data-autofocus]");
    if (first) {
      first.focus();
      if (first instanceof HTMLInputElement) first.select();
    }
    const onCancel = (e: Event) => {
      e.preventDefault(); // we decide when to close (not while saving)
      if (!busyRef.current) onCloseRef.current();
    };
    dialog.addEventListener("cancel", onCancel);
    return () => {
      dialog.removeEventListener("cancel", onCancel);
      if (dialog.open) dialog.close();
      restoreFocus(returnTo);
    };
  }, []);

  return (
    <dialog
      ref={ref}
      className={`dialog dialog-${size}`}
      role={role === "alertdialog" ? "alertdialog" : undefined}
      aria-labelledby={titleId}
      aria-describedby={description ? descId : undefined}
      onMouseDown={(e) => {
        pressedBackdrop.current = e.target === e.currentTarget;
      }}
      onClick={(e) => {
        if (pressedBackdrop.current && e.target === e.currentTarget && !busy) onClose();
        pressedBackdrop.current = false;
      }}
    >
      <div className="dialog-body">
        <div className="dialog-head">
          <h2 id={titleId}>{title}</h2>
          <button type="button" className="icon-btn" onClick={onClose} disabled={busy} aria-label="Close">
            <Icon name="close" />
          </button>
        </div>
        {description && (
          <div id={descId} className="dialog-desc">
            {description}
          </div>
        )}
        {children}
      </div>
    </dialog>
  );
}
