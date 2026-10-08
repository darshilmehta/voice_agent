import { AppShell } from "@/components/AppShell";
import { backendUrl } from "@/lib/backend-url";

// BACKEND_URL is read per request, not baked in at build time.
export const dynamic = "force-dynamic";

export default function Home() {
  return <AppShell backendUrl={backendUrl()} />;
}
