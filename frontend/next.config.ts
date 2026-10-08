import type { NextConfig } from "next";

const config: NextConfig = {
  // Self-contained server bundle for the container image (infra/docker).
  output: "standalone",
  reactStrictMode: true,
  poweredByHeader: false,
};

export default config;
