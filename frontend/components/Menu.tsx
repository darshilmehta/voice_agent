"use client";

/**
 * A "⋯" menu button (WAI-ARIA menu button pattern). Arrow keys, Home/End move between items, Enter/Space choose,
 * Esc or Tab close and return focus to the button. The popup is portalled to <body> with fixed positioning so the
 * scrolling sidebar can't clip it.
 */

import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState, type KeyboardEvent } from "react";
import { createPortal } from "react-dom";

import { Icon, type IconName } from "./Icon";

export interface MenuItem {
  /** Stable key, also lets callers pick a subset ("rename", "pin", "archive", "delete"). */
  id: string;
  label: string;
  icon?: IconName;
  onSelect: () => void;
  danger?: boolean;
  /** Draw a separator above this item. */
  separated?: boolean;
}

interface MenuProps {
  /** Accessible name of the button, e.g. "Actions for Annual report". */
  label: string;
  items: MenuItem[];
  className?: string;
  /** Show the button's icon at this size. */
  iconSize?: number;
}

const GAP = 4;
const EDGE = 8;

export function Menu({ label, items, className = "icon-btn", iconSize = 16 }: MenuProps) {
  const [open, setOpen] = useState(false);
  const [focusIndex, setFocusIndex] = useState(0);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const itemRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const menuId = useId();

  const close = useCallback((returnFocus: boolean) => {
    setOpen(false);
    setPos(null);
    if (returnFocus) buttonRef.current?.focus();
  }, []);

  const openAt = (index: number) => {
    setFocusIndex(index);
    setOpen(true);
  };

  // Place the popup under the button (right-aligned), flipping above when there is no room below.
  useLayoutEffect(() => {
    if (!open) return;
    const button = buttonRef.current;
    const menu = menuRef.current;
    if (!button || !menu) return;
    const b = button.getBoundingClientRect();
    const { offsetWidth: w, offsetHeight: h } = menu;
    let left = b.right - w;
    if (left < EDGE) left = Math.min(b.left, window.innerWidth - w - EDGE);
    let top = b.bottom + GAP;
    if (top + h > window.innerHeight - EDGE && b.top - GAP - h > EDGE) top = b.top - GAP - h;
    setPos({ top: Math.round(top), left: Math.round(Math.max(EDGE, left)) });
  }, [open]);

  useEffect(() => {
    if (open && pos) itemRefs.current[focusIndex]?.focus();
  }, [open, pos, focusIndex]);

  // Close on outside press, scroll or resize.
  useEffect(() => {
    if (!open) return;
    const onPointer = (e: PointerEvent) => {
      const t = e.target as Node;
      if (!menuRef.current?.contains(t) && !buttonRef.current?.contains(t)) close(false);
    };
    const onScroll = (e: Event) => {
      if (!menuRef.current?.contains(e.target as Node)) close(false);
    };
    const onResize = () => close(false);
    document.addEventListener("pointerdown", onPointer, true);
    window.addEventListener("scroll", onScroll, true);
    window.addEventListener("resize", onResize);
    return () => {
      document.removeEventListener("pointerdown", onPointer, true);
      window.removeEventListener("scroll", onScroll, true);
      window.removeEventListener("resize", onResize);
    };
  }, [open, close]);

  const onMenuKey = (e: KeyboardEvent) => {
    const n = items.length;
    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        setFocusIndex((i) => (i + 1) % n);
        break;
      case "ArrowUp":
        e.preventDefault();
        setFocusIndex((i) => (i - 1 + n) % n);
        break;
      case "Home":
        e.preventDefault();
        setFocusIndex(0);
        break;
      case "End":
        e.preventDefault();
        setFocusIndex(n - 1);
        break;
      case "Escape":
        e.preventDefault();
        e.stopPropagation();
        close(true);
        break;
      case "Tab":
        e.preventDefault();
        close(true);
        break;
    }
  };

  const choose = (item: MenuItem) => {
    // Focus goes back to the button first, so a dialog opened by the item returns focus there when it closes.
    close(true);
    item.onSelect();
  };

  return (
    <>
      <button
        ref={buttonRef}
        type="button"
        className={className}
        aria-label={label}
        title={label}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        data-open={open || undefined}
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          if (open) close(false);
          else openAt(0);
        }}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown" || e.key === "ArrowUp") {
            e.preventDefault();
            openAt(e.key === "ArrowUp" ? items.length - 1 : 0);
          }
        }}
      >
        <Icon name="more" size={iconSize} />
      </button>
      {open &&
        createPortal(
          <div
            ref={menuRef}
            id={menuId}
            role="menu"
            aria-label={label}
            className="menu"
            style={pos ? { top: pos.top, left: pos.left } : { top: 0, left: 0, visibility: "hidden" }}
            onKeyDown={onMenuKey}
          >
            {items.map((item, i) => (
              <div key={item.id} role="none" className={item.separated ? "menu-sep" : undefined}>
                <button
                  ref={(el) => {
                    itemRefs.current[i] = el;
                  }}
                  type="button"
                  role="menuitem"
                  tabIndex={i === focusIndex ? 0 : -1}
                  className={item.danger ? "menu-item danger" : "menu-item"}
                  onClick={() => choose(item)}
                  onMouseMove={() => {
                    if (focusIndex !== i) setFocusIndex(i);
                  }}
                >
                  {item.icon && <Icon name={item.icon} />}
                  {item.label}
                </button>
              </div>
            ))}
          </div>,
          document.body,
        )}
    </>
  );
}
