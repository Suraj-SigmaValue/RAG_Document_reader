"use client";

import { FormEvent, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

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
  type?: "text" | "table" | "image";
  content?: string;
  image_base64?: string;
  image_mime?: string;
  relevance_score?: number;
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

function MarkdownAnswer({ content }: { content: string }) {
  return (
    <div className="answer-markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          img: ({ src, alt }) => {
            if (!src) {
              return null;
            }

            return <img src={src} alt={alt || ""} />;
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function chunkImageSrc(chunk: Chunk) {
  if (!chunk.image_base64) {
    return "";
  }

  return `data:${chunk.image_mime || "image/png"};base64,${chunk.image_base64}`;
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

  const answerImages = useMemo(() => {
    return askResult?.chunks.filter((chunk) => chunk.type === "image" && chunk.image_base64) || [];
  }, [askResult]);

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

              {answerImages.length ? (
                <section className="visual-references" aria-label="Visual references">
                  <h3>Supporting Images</h3>
                  <div className="image-grid">
                    {answerImages.map((chunk, index) => (
                      <figure key={`${chunk.page}-${index}`} className="answer-image">
                        <img
                          src={chunkImageSrc(chunk)}
                          alt={`Supporting image from page ${chunk.page}`}
                        />
                        <figcaption>
                          {chunk.source} · Page {chunk.page}
                        </figcaption>
                      </figure>
                    ))}
                  </div>
                </section>
              ) : null}

              <details>
                <summary>Retrieved Source Chunks</summary>
                <div className="chunks">
                  {askResult.chunks.map((chunk, index) => (
                    <div key={index} className="chunk-card">
                      <div className="chunk-meta">
                        <span className="chunk-source">{chunk.source}</span>
                        <span className="chunk-page">Page {chunk.page}</span>
                      </div>
                      {chunk.type === "image" && chunk.image_base64 ? (
                        <figure className="chunk-image">
                          <img
                            src={chunkImageSrc(chunk)}
                            alt={`Retrieved image from page ${chunk.page}`}
                          />
                        </figure>
                      ) : (
                        <p className="chunk-content">{chunk.content}</p>
                      )}
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
