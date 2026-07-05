import type { NextConfig } from "next";
import path from "node:path";

const nextConfig: NextConfig = {
  // A stray package-lock.json one level up (outside this project) made
  // Next.js infer the wrong workspace root. Pin it explicitly to this
  // directory to silence the warning.
  turbopack: {
    root: path.resolve(__dirname),
  },
};

export default nextConfig;
