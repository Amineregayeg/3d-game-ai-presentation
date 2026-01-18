import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
});

export const metadata: Metadata = {
  title: "AASP - AI Agent Security Platform",
  description: "Secure what AI agents do, not just what they can access. The first real-time control plane for AI agent actions.",
  keywords: ["AI security", "agent security", "MCP", "LLM security", "AI governance"],
  authors: [{ name: "AASP Team" }],
  openGraph: {
    title: "AASP - AI Agent Security Platform",
    description: "Secure what AI agents do, not just what they can access.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} font-sans antialiased`}>
        {children}
      </body>
    </html>
  );
}
