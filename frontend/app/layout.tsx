import type { Metadata, Viewport } from "next";
import { connection } from "next/server";
import type { ReactNode } from "react";

import { AppProviders } from "@/components/AppProviders";
import { backendUrl } from "@/lib/backend-url";
import { PRODUCT_NAME, TAGLINE } from "@/lib/brand";

import "./globals.css";
import "./styles/shell.css";
import "./styles/overlays.css";
import "./styles/pages.css";
import "./styles/chat.css";
import "./styles/summary.css";
import "./styles/voice.css";
import "./styles/canvas.css";

// BACKEND_URL is read per request on the server and handed to the client, never baked in at build time
// (docs/DESIGN.md §6.5). Every route renders under this layout, so every route is dynamic.
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  applicationName: PRODUCT_NAME,
  title: { default: `${PRODUCT_NAME} · ${TAGLINE}`, template: `%s · ${PRODUCT_NAME}` },
  description: `${TAGLINE}: ask about your PDFs, Word and PowerPoint files out loud, in English or Hindi, fully on this machine.`,
  openGraph: {
    type: "website",
    siteName: PRODUCT_NAME,
    title: `${PRODUCT_NAME} · ${TAGLINE}`,
    description: TAGLINE,
  },
  twitter: { card: "summary", title: `${PRODUCT_NAME} · ${TAGLINE}`, description: TAGLINE },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f7f7f5" },
    { media: "(prefers-color-scheme: dark)", color: "#141413" },
  ],
};

export default async function RootLayout({ children }: { children: ReactNode }) {
  await connection();
  return (
    <html lang="en">
      <body>
        <AppProviders backendUrl={backendUrl()}>{children}</AppProviders>
      </body>
    </html>
  );
}
