import type { Metadata } from "next";
import "../styles/globals.css";

export const metadata: Metadata = {
  title: "User Input Data Agent",
  description: "RAG framework to deal with User Input Data Agent.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
