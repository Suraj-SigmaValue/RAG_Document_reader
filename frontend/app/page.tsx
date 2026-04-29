"use client";

import { FormEvent, useMemo, useState } from "react";

type UploadResult = {
  document_name: string;
  pages_or_sections: number;
  chunk_count: number;
  message: string;
  token_usage: TokenUsage;
};

type Chunk = {
  source: string;
  page: string;
  content: string;
};

type AskResult = {
  answer: string;
  chunks: Chunk[];
  token_usage: TokenUsage;
};

type TokenUsage = {
  input: number;
  output: number;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ||
  "http://localhost:8080";

function formatInlineMarkdown(text: string) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }

    return part;
  });
}

function MarkdownAnswer({ content }: { content: string }) {
  const blocks = content
    .replace(/\r\n/g, "\n")
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean);

  return (
    <div className="answer-markdown">
      {blocks.map((block, index) => {
        const heading = block.match(/^(#{1,6})\s+(.+)$/);
        if (heading) {
          const level = Math.min(heading[1].length + 2, 4);
          const HeadingTag = `h${level}` as keyof JSX.IntrinsicElements;
          return <HeadingTag key={index}>{formatInlineMarkdown(heading[2])}</HeadingTag>;
        }

        const lines = block.split("\n").map((line) => line.trim()).filter(Boolean);
        const allListItems = lines.every((line) => /^(\d+\.\s+|-+\s+)/.test(line));

        if (allListItems) {
          const ordered = lines.every((line) => /^\d+\.\s+/.test(line));
          const ListTag = ordered ? "ol" : "ul";
          return (
            <ListTag key={index}>
              {lines.map((line, lineIndex) => (
                <li key={lineIndex}>
                  {formatInlineMarkdown(line.replace(/^(\d+\.\s+|-+\s+)/, ""))}
                </li>
              ))}
            </ListTag>
          );
        }

        return (
          <p key={index}>
            {lines.map((line, lineIndex) => (
              <span key={lineIndex}>
                {formatInlineMarkdown(line)}
                {lineIndex < lines.length - 1 ? <br /> : null}
              </span>
            ))}
          </p>
        );
      })}
    </div>
  );
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("");
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const [askResult, setAskResult] = useState<AskResult | null>(null);
  const [busy, setBusy] = useState<"upload" | "ask" | null>(null);
  const [error, setError] = useState("");

  const tokenUsage = useMemo<TokenUsage>(() => {
    return askResult?.token_usage || uploadResult?.token_usage || { input: 0, output: 0 };
  }, [askResult, uploadResult]);

  async function parseError(response: Response) {
    try {
      const data = await response.json();
      return data.detail || "Request failed.";
    } catch {
      return "Request failed.";
    }
  }

  async function uploadDocument(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) {
      setError("Choose a PDF or DOCX file first.");
      return;
    }

    setBusy("upload");
    setError("");
    setAskResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE_URL}/documents`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(await parseError(response));
      }

      setUploadResult(await response.json());
    } catch (err) {
      setError(err instanceof Error ? `${err.message} (${API_BASE_URL})` : `Upload failed. (${API_BASE_URL})`);
    } finally {
      setBusy(null);
    }
  }

  async function askQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!question.trim()) {
      setError("Enter a question.");
      return;
    }

    setBusy("ask");
    setError("");

    try {
      const response = await fetch(`${API_BASE_URL}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        throw new Error(await parseError(response));
      }

      setAskResult(await response.json());
    } catch (err) {
      setError(err instanceof Error ? `${err.message} (${API_BASE_URL})` : `Question failed. (${API_BASE_URL})`);
    } finally {
      setBusy(null);
    }
  }

  return (
    <main className="shell">
      <section className="topbar">
        <div>
          <h1>RAG Document Reader</h1>
          <p>Upload a PDF or DOCX, then ask questions against hybrid FAISS and BM25 retrieval.</p>
        </div>
        <div className="tokens" aria-label="Token usage">
          <span>Input tokens: {tokenUsage.input}</span>
          <span>Output tokens: {tokenUsage.output}</span>
        </div>
      </section>

      {error ? <div className="alert">{error}</div> : null}

      <section className="workspace">
        <aside className="panel">
          <form onSubmit={uploadDocument} className="stack">
            <label htmlFor="document">Document</label>
            <input
              id="document"
              type="file"
              accept=".pdf,.docx"
              onChange={(event) => setFile(event.target.files?.[0] || null)}
            />
            <button type="submit" disabled={busy === "upload"}>
              {busy === "upload" ? "Processing..." : "Process document"}
            </button>
          </form>

          {uploadResult ? (
            <dl className="stats">
              <div>
                <dt>File</dt>
                <dd>{uploadResult.document_name}</dd>
              </div>
              <div>
                <dt>Pages/sections</dt>
                <dd>{uploadResult.pages_or_sections}</dd>
              </div>
              <div>
                <dt>Chunks</dt>
                <dd>{uploadResult.chunk_count}</dd>
              </div>
            </dl>
          ) : null}
        </aside>

        <section className="panel reader">
          <form onSubmit={askQuestion} className="ask">
            <label htmlFor="question">Question</label>
            <textarea
              id="question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask about definitions, clauses, calculations, or document conclusions..."
              rows={4}
            />
            <button type="submit" disabled={busy === "ask" || !uploadResult}>
              {busy === "ask" ? "Thinking..." : "Get answer"}
            </button>
          </form>

          {askResult ? (
            <div className="answer">
              <h2>Answer</h2>
              <MarkdownAnswer content={askResult.answer} />

              <details>
                <summary>Retrieved Source Chunks</summary>
                <div className="chunks">
                  {askResult.chunks.map((chunk, index) => (
                    <div key={index} className="chunk-card">
                      <div className="chunk-meta">
                        <span className="chunk-source">{chunk.source}</span>
                        <span className="chunk-page">Page {chunk.page}</span>
                      </div>
                      <p className="chunk-content">{chunk.content}</p>
                    </div>
                  ))}
                </div>
              </details>
            </div>
          ) : (
            <div className="empty">Process a document to start asking questions.</div>
          )}
        </section>
      </section>
    </main>
  );
}
