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
          table: ({ children, ...props }) => (
            <div className="table-frame">
              <table {...props}>{children}</table>
            </div>
          ),
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

type GraphNodeId = "start" | "retrieve" | "generate" | "end";
type PipelineDurations = Partial<Record<GraphNodeId, number>>;

function formatDuration(ms?: number | null) {
  if (ms == null) {
    return "";
  }

  const seconds = ms / 1000;
  if (seconds < 1) {
    return "< 1s";
  }

  return `${seconds.toFixed(seconds >= 10 ? 0 : 1)}s`;
}

function PipelineGraph({
  active,
  ready,
  durations,
  totalDuration,
}: {
  active: GraphNodeId | null;
  ready: boolean;
  durations: PipelineDurations;
  totalDuration: number | null;
}) {
  const nodes: { id: GraphNodeId; label: string; sub?: string }[] = [
    { id: "start", label: "Start", sub: "Document ready" },
    { id: "retrieve", label: "Retrieve", sub: "FAISS / BM25 / Rerank" },
    { id: "generate", label: "Generate", sub: "gpt-4o-mini / temperature 0.2" },
    { id: "end", label: "Complete", sub: "Answer delivered" },
  ];
  const activeIndex = active ? nodes.findIndex((node) => node.id === active) : -1;

  return (
    <div className="pipeline-graph" aria-label="LangGraph execution pipeline">
      <div className="pipeline-heading">
        <p className="pipeline-title">LangGraph Pipeline</p>
        <span className="pipeline-state">
          {totalDuration ? `Total ${formatDuration(totalDuration)}` : active ? "Running" : ready ? "Ready" : "Idle"}
        </span>
      </div>
      <div className="pipeline-nodes">
        {nodes.map((node, index) => {
          const isActive = active === node.id;
          const duration = durations[node.id];
          const isComplete = duration != null || activeIndex > index;
          const isReadyStart = !active && ready && node.id === "start";
          const status = isActive && !isComplete ? "In progress" : isComplete ? "Complete" : isReadyStart ? "Ready" : "Waiting";

          return (
            <div
              key={node.id}
              className={[
                "pipeline-step",
                isActive ? "pipeline-step--active" : "",
                isComplete ? "pipeline-step--complete" : "",
                isReadyStart ? "pipeline-step--ready" : "",
              ]
                .filter(Boolean)
                .join(" ")}
            >
              <span className="pipeline-marker" aria-hidden="true">
                {isComplete ? "" : index + 1}
              </span>
              <div
                className={[
                  "pipeline-node",
                  isActive ? "pipeline-node--active" : "",
                  isComplete ? "pipeline-node--complete" : "",
                  isReadyStart ? "pipeline-node--ready" : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
              >
                <span className="pipeline-node-topline">
                  <span className="pipeline-node-label">{node.label}</span>
                  <span className="pipeline-node-status">
                    {duration != null ? formatDuration(duration) : status}
                  </span>
                </span>
                <span className="pipeline-node-sub">{node.sub}</span>
              </div>
            </div>
          );
        })}
      </div>
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
  const [activeNode, setActiveNode] = useState<GraphNodeId | null>(null);
  const [stageDurations, setStageDurations] = useState<PipelineDurations>({});
  const [totalDuration, setTotalDuration] = useState<number | null>(null);

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
    setActiveNode(null);
    setStageDurations({});
    setTotalDuration(null);

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
    setStageDurations({});
    setTotalDuration(null);
    setActiveNode("start");

    try {
      const totalStartedAt = performance.now();
      const retrieveStartedAt = performance.now();
      setStageDurations((prev) => ({ ...prev, start: retrieveStartedAt - totalStartedAt }));
      setActiveNode("retrieve");

      const response = await fetch(`${API_BASE_URL}/ask/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        throw new Error(await parseError(response));
      }

      const generateStartedAt = performance.now();
      setStageDurations((prev) => ({ ...prev, retrieve: generateStartedAt - retrieveStartedAt }));

      const reader = response.body?.getReader();
      const decoder = new TextDecoder("utf-8");

      if (!reader) throw new Error("No response body");

      setActiveNode("generate");
      let fullAnswer = "";

      setAskResult({ answer: "", chunks: [], token_usage: { input: 0, output: 0 } });

      let buffer = "";
      let completed = false;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        let boundary = buffer.indexOf("\n\n");
        while (boundary !== -1) {
          const message = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);

          const lines = message.split("\n");
          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const dataStr = line.replace("data: ", "").trim();
              if (!dataStr) continue;

              try {
                const data = JSON.parse(dataStr);
                if (data.type === "token") {
                  fullAnswer += data.content;
                  setAskResult((prev) => prev ? { ...prev, answer: fullAnswer } : null);
                } else if (data.type === "done") {
                  const completeStartedAt = performance.now();
                  setAskResult({
                    answer: fullAnswer,
                    chunks: data.chunks,
                    token_usage: data.token_usage,
                  });
                  const completedAt = performance.now();
                  setStageDurations((prev) => ({
                    ...prev,
                    generate: completeStartedAt - generateStartedAt,
                    end: completedAt - completeStartedAt,
                  }));
                  setTotalDuration(completedAt - totalStartedAt);
                  setActiveNode("end");
                  completed = true;
                }
              } catch (e) {
                console.error("Failed to parse SSE line", dataStr);
              }
            }
          }

          boundary = buffer.indexOf("\n\n");
        }
      }

      if (!completed) {
        const completedAt = performance.now();
        setStageDurations((prev) => ({ ...prev, generate: completedAt - generateStartedAt, end: 0 }));
        setTotalDuration(completedAt - totalStartedAt);
        setActiveNode("end");
      }
    } catch (err) {
      setError(err instanceof Error ? `${err.message} (${API_BASE_URL})` : `Question failed. (${API_BASE_URL})`);
      setActiveNode(null);
    } finally {
      setBusy(null);
    }
  }

  return (
    <main className="shell">
      <section className="topbar">
        <div className="logo-title">
          <img src="/DS.png" alt="Logo" className="logo" />

          <div className="title-block">
            <h1>User Input Data Agent</h1>
            <p>
              Upload a PDF or DOCX, then ask questions against hybrid FAISS and BM25 retrieval.
            </p>
          </div>
        </div>

        <div className="tokens">
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

          <PipelineGraph
            active={activeNode}
            ready={Boolean(uploadResult)}
            durations={stageDurations}
            totalDuration={totalDuration}
          />
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