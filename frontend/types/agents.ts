export type GraphNodeId = "start" | "retrieve" | "generate" | "end";

export type PipelineDurations = Partial<Record<GraphNodeId, number>>;
