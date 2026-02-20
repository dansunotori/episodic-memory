// No top-level import of @xenova/transformers — loaded lazily to avoid
// multi-GB ONNX runtime loading in MCP server processes that never search.

let embeddingPipeline: any = null;
let embeddingCallCount = 0;

// How often to force V8 GC during batch embedding. V8 doesn't know about
// native ONNX tensor memory, so without periodic GC the small JS wrappers
// pile up and anchor large native allocations until the process OOMs.
const GC_EVERY_N_CALLS = 50;

export async function initEmbeddings(): Promise<void> {
  if (!embeddingPipeline) {
    console.log('Loading embedding model (first run may take time)...');
    const { pipeline } = await import('@xenova/transformers');
    embeddingPipeline = await pipeline(
      'feature-extraction',
      'Xenova/all-MiniLM-L6-v2'
    );
    embeddingCallCount = 0;
    console.log('Embedding model loaded');
  }
}

export async function resetEmbeddings(): Promise<void> {
  if (embeddingPipeline) {
    // Pipeline.dispose() releases the ONNX InferenceSession's native C++
    // memory. Without this, nulling the reference alone leaves the native
    // session allocated until (if ever) V8 GC collects the JS wrapper.
    await embeddingPipeline.dispose();
    embeddingPipeline = null;
    embeddingCallCount = 0;
  }
}

export async function generateEmbedding(text: string): Promise<number[]> {
  if (!embeddingPipeline) {
    await initEmbeddings();
  }

  // Truncate text to avoid token limits (512 tokens max for this model)
  const truncated = text.substring(0, 2000);

  const output = await embeddingPipeline!(truncated, {
    pooling: 'mean',
    normalize: true
  });

  const embedding = Array.from(output.data) as number[];

  // Release the tensor's native memory if dispose() is available.
  if (typeof output.dispose === 'function') {
    output.dispose();
  }

  embeddingCallCount++;

  // Periodically force GC to reclaim native ONNX tensor memory. V8 only
  // tracks JS heap pressure and won't GC aggressively enough on its own
  // when each ~200-byte JS wrapper anchors a much larger native allocation.
  if (embeddingCallCount % GC_EVERY_N_CALLS === 0 && typeof globalThis.gc === 'function') {
    globalThis.gc();
  }

  return embedding;
}

export async function generateExchangeEmbedding(
  userMessage: string,
  assistantMessage: string,
  toolNames?: string[]
): Promise<number[]> {
  // Combine user question, assistant answer, and tools used for better searchability
  let combined = `User: ${userMessage}\n\nAssistant: ${assistantMessage}`;

  // Include tool names in embedding for tool-based searches
  if (toolNames && toolNames.length > 0) {
    combined += `\n\nTools: ${toolNames.join(', ')}`;
  }

  return generateEmbedding(combined);
}
