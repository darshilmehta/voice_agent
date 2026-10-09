"use client";

import { useRef, type ReactNode } from "react";

/**
 * A chat's title, text that fades in when it changes (an automatic title arriving, a regenerated one, a rename) so the
 * header and the sidebar don't jump. The title a component first saw doesn't animate, so lists don't fade on load.
 * `children` can stand in for the plain text (the sidebar highlights search matches).
 */
export function TitleText({ title, children }: { title: string; children?: ReactNode }) {
  const first = useRef(title);
  return (
    <span key={title} className={title === first.current ? undefined : "title-swap"}>
      {children ?? title}
    </span>
  );
}
