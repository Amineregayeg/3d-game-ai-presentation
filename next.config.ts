import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // basePath: "/3dgameassistant",  // Disabled for local development
  // assetPrefix: "/3dgameassistant",

  // Fix turbopack workspace root detection
  turbopack: {
    root: process.cwd(),
  },

  // Increase body size limit for audio uploads (STT)
  experimental: {
    serverActions: {
      bodySizeLimit: "50mb",
    },
  },

  async rewrites() {
    return [
      // Proxy Flask backend routes (exclude Next.js API routes)
      {
        source: "/api/tasks/:path*",
        destination: "http://localhost:5000/api/tasks/:path*",
      },
      {
        source: "/api/team/:path*",
        destination: "http://localhost:5000/api/team/:path*",
      },
      {
        source: "/api/activity/:path*",
        destination: "http://localhost:5000/api/activity/:path*",
      },
      {
        source: "/api/milestones/:path*",
        destination: "http://localhost:5000/api/milestones/:path*",
      },
      {
        source: "/api/decisions/:path*",
        destination: "http://localhost:5000/api/decisions/:path*",
      },
      {
        source: "/api/resources/:path*",
        destination: "http://localhost:5000/api/resources/:path*",
      },
      {
        source: "/api/glossary/:path*",
        destination: "http://localhost:5000/api/glossary/:path*",
      },
      {
        source: "/api/changelog/:path*",
        destination: "http://localhost:5000/api/changelog/:path*",
      },
      {
        source: "/api/vault/:path*",
        destination: "http://localhost:5000/api/vault/:path*",
      },
      {
        source: "/api/context/:path*",
        destination: "http://localhost:5000/api/context/:path*",
      },
      {
        source: "/api/health/:path*",
        destination: "http://localhost:5000/api/health/:path*",
      },
      {
        source: "/api/rag/:path*",
        destination: "http://localhost:5000/api/rag/:path*",
      },
      {
        source: "/api/avatar/:path*",
        destination: "http://localhost:5000/api/avatar/:path*",
      },
      {
        source: "/api/blender/:path*",
        destination: "http://localhost:5000/api/blender/:path*",
      },
      {
        source: "/api/gpu/:path*",
        destination: "http://localhost:5000/api/gpu/:path*",
      },
      {
        source: "/api/voxformer/:path*",
        destination: "http://localhost:5000/api/voxformer/:path*",
      },
      {
        source: "/static/:path*",
        destination: "http://localhost:5000/static/:path*",
      },
      // Note: /api/salesforce/* and /api/elevenlabs/* are handled by Next.js API routes
    ];
  },
};

export default nextConfig;
