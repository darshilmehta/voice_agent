"use client";

/**
 * Sidebar + main column. Below 820 px the sidebar becomes a modal drawer: a toggle in the top bar opens it, Esc,
 * the scrim or the close button close it, the rest of the page is inert while it is open, and navigating closes it.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useCallback, useEffect, useRef, useState, useSyncExternalStore, type ReactNode } from "react";

import { useBackend } from "@/lib/backend-context";

import { Icon } from "./Icon";
import { Sidebar } from "./Sidebar";

const NARROW = "(max-width: 819.98px)";

function useMediaQuery(query: string): boolean {
  return useSyncExternalStore(
    (onChange) => {
      const mql = window.matchMedia(query);
      mql.addEventListener("change", onChange);
      return () => mql.removeEventListener("change", onChange);
    },
    () => window.matchMedia(query).matches,
    () => false,
  );
}

export function AppFrame({ children }: { children: ReactNode }) {
  const { appTitle } = useBackend();
  const pathname = usePathname();
  const narrow = useMediaQuery(NARROW);
  const [open, setOpen] = useState(false);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const sidebarRef = useRef<HTMLElement>(null);
  /** Where focus goes when the drawer closes: back to the toggle, or to the new page's content. */
  const focusAfterClose = useRef<"toggle" | "main" | null>(null);

  const modal = narrow && open;

  const close = useCallback((focus: "toggle" | "main" = "toggle") => {
    focusAfterClose.current = focus;
    setOpen(false);
  }, []);

  // Navigating closes the drawer and lands focus on the new page.
  const lastPath = useRef(pathname);
  useEffect(() => {
    if (lastPath.current === pathname) return;
    lastPath.current = pathname;
    if (open) close("main");
  }, [pathname, open, close]);

  // Growing past the breakpoint turns the drawer back into a plain sidebar.
  useEffect(() => {
    if (!narrow && open) setOpen(false);
  }, [narrow, open]);

  useEffect(() => {
    if (modal) {
      sidebarRef.current?.querySelector<HTMLElement>("[data-drawer-focus]")?.focus();
      const onKey = (e: KeyboardEvent) => {
        if (e.key === "Escape" && !e.defaultPrevented && !document.querySelector("dialog[open], [role=menu]")) {
          close("toggle");
        }
      };
      document.addEventListener("keydown", onKey);
      return () => document.removeEventListener("keydown", onKey);
    }
    const target = focusAfterClose.current;
    focusAfterClose.current = null;
    if (target === "toggle") toggleRef.current?.focus();
    else if (target === "main") document.getElementById("main")?.focus({ preventScroll: true });
  }, [modal, close]);

  const requestOpen = useCallback(() => {
    if (window.matchMedia(NARROW).matches) setOpen(true);
  }, []);

  return (
    <div className="app" data-drawer={modal ? "open" : undefined}>
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      <aside
        ref={sidebarRef}
        id="sidebar"
        className="sidebar"
        data-open={open || undefined}
        aria-label="Sidebar"
        role={modal ? "dialog" : undefined}
        aria-modal={modal || undefined}
      >
        <Sidebar onClose={narrow ? () => close("toggle") : undefined} onRequestOpen={requestOpen} searchRef={searchRef} />
      </aside>
      {modal && <div className="scrim" onClick={() => close("toggle")} aria-hidden />}

      <div className="main-col" inert={modal || undefined}>
        <header className="topbar">
          <button
            ref={toggleRef}
            type="button"
            className="icon-btn"
            aria-label="Open sidebar"
            aria-expanded={open}
            aria-controls="sidebar"
            onClick={() => setOpen(true)}
          >
            <Icon name="menu" size={18} />
          </button>
          <Link href="/" className="topbar-title">
            {appTitle}
          </Link>
        </header>
        <main id="main" className="main" tabIndex={-1}>
          {children}
        </main>
      </div>
    </div>
  );
}
