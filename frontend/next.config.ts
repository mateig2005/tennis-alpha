import type { NextConfig } from "next";

// The Python API (FastAPI, src/api.py). The browser only ever talks to this site: /api/* is forwarded server-side,
// so the API can stay bound to localhost and no cross-origin setup is needed.
const API_URL = process.env.TENNIS_API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/:path*` }];
  },
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
