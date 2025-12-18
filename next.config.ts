import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  basePath: "/3dgameassistant",
  assetPrefix: "/3dgameassistant",

  // Increase body size limit for audio uploads (STT)
  experimental: {
    serverActions: {
      bodySizeLimit: "50mb",
    },
  },

  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:5000/api/:path*",
      },
      {
        source: "/static/:path*",
        destination: "http://localhost:5000/static/:path*",
      },
    ];
  },
};

export default nextConfig;
