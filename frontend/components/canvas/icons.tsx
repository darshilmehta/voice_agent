/** A few stroke icons the canvas needs that the app's icon set doesn't have (same 24×24 grid and weight as ../Icon). */

import type { SVGProps } from "react";

const PATHS = {
  pin: (
    <>
      <path d="M9 4h6l-.8 5.2 3.3 3.3H6.5l3.3-3.3z" />
      <path d="M12 12.5V20" />
    </>
  ),
  table: (
    <>
      <rect x="3.5" y="5" width="17" height="14" rx="2" />
      <path d="M3.5 10h17M9.5 10v9M15 10v9" />
    </>
  ),
  chart: (
    <>
      <path d="M4.5 4.5v15h15" />
      <path d="M8.5 15v-3.5M12.5 15V8.5M16.5 15v-5" />
    </>
  ),
  grip: (
    <>
      <circle cx="9" cy="7" r="1.2" fill="currentColor" stroke="none" />
      <circle cx="15" cy="7" r="1.2" fill="currentColor" stroke="none" />
      <circle cx="9" cy="12" r="1.2" fill="currentColor" stroke="none" />
      <circle cx="15" cy="12" r="1.2" fill="currentColor" stroke="none" />
      <circle cx="9" cy="17" r="1.2" fill="currentColor" stroke="none" />
      <circle cx="15" cy="17" r="1.2" fill="currentColor" stroke="none" />
    </>
  ),
  formula: <path d="M15.5 5.5c-1.6-.7-3 .2-3.4 2L10.6 14c-.4 1.8-1.8 2.7-3.4 2M8.5 10h6" />,
  sheet: <path d="M5 8.5c0-1 .8-1.8 1.8-1.8h10.4c1 0 1.8.8 1.8 1.8V19H5zM9 4.5h6" />,
} as const;

export type CanvasIconName = keyof typeof PATHS;

export function CanvasIcon({
  name,
  size = 16,
  filled = false,
  className,
  ...rest
}: { name: CanvasIconName; size?: number; filled?: boolean } & Omit<SVGProps<SVGSVGElement>, "name">) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill={filled ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      className={className ? `icon ${className}` : "icon"}
      {...rest}
    >
      {PATHS[name]}
    </svg>
  );
}
