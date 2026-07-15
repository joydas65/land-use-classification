import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TerraClass",
  description: "A leakage-aware satellite land-use classifier.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
