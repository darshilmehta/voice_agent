"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { useHealth } from "@/lib/health";

/** The sidebar's footer: overall health from GET /health; opens the system status page. */
export function HealthIndicator() {
  const { report, error, checking } = useHealth();
  const pathname = usePathname();

  let tone: "ok" | "degraded" | "down" | "idle" = "idle";
  let label = "Checking system…";
  let detail: string | null = null;
  if (report) {
    const attention = report.providers.filter((p) => p.status !== "ok" && p.status !== "disabled").length;
    if (report.status === "ok") {
      tone = "ok";
      label = "All systems OK";
    } else {
      tone = "degraded";
      label = "Needs attention";
      detail = `${attention} of ${report.providers.length}`;
    }
  } else if (error && !checking) {
    tone = "down";
    label = "Backend unreachable";
  }

  return (
    <Link
      href="/status"
      className="sb-health"
      aria-current={pathname === "/status" ? "page" : undefined}
      title="System status"
    >
      <span className={`dot ${tone}`} aria-hidden />
      <span className="sb-health-label">
        {label}
        <span className="visually-hidden"> — open system status</span>
      </span>
      {detail && <span className="sb-health-detail">{detail}</span>}
    </Link>
  );
}
