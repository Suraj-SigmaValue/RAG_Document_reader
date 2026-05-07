export type TokenUsage = {
  input: number;
  output: number;
};

export type UploadResult = {
  document_name: string;
  pages_or_sections: number;
  chunk_count: number;
  message: string;
  token_usage: TokenUsage;
};

export type Chunk = {
  source: string;
  page: string;
  type?: "text" | "table" | "image";
  content?: string;
  image_base64?: string;
  image_mime?: string;
  relevance_score?: number;
};

export type AskResult = {
  answer: string;
  chunks: Chunk[];
  token_usage: TokenUsage;
};
