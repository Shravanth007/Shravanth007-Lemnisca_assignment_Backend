# Written Answers

---

## Q1 -- Routing Logic

The router classifies a query as **complex** if any of the following conditions are true:

1. The word count is 12 or more.
2. The query contains analytical keywords: `compare`, `difference`, `explain`, `why`, `how does`, `how do`, `how can`, `how should`, `pros and cons`, `trade-off`, `recommend`, `evaluate`, `analyse`, `step by step`, `in detail`, `walk me through`, `advantages`, `disadvantages`.
3. The query contains two or more question marks (multi-part question).
4. The query contains subordinate clause markers: `because`, `although`, `however`, `whereas`, `nevertheless`, `furthermore`, `moreover`, `consequently`, or an `if...then` pattern.

If none of these match, the query is classified as **simple**. Simple queries go to `llama-3.1-8b-instant` and complex queries go to `llama-3.3-70b-versatile`.

**Examples -- Simple (routed to 8B model):**
- "What is the price?" -- Short, no keyword triggers, single question mark. Classified as simple.
- "Keyboard shortcuts" -- Two words, no triggers at all. Classified as simple.

**Examples -- Complex (routed to 70B model):**
- "Compare the Pro and Enterprise plans and explain the differences in features and pricing" -- Triggers on "compare", "explain", "differences", and word count is well above 12.
- "What integrations are available? And how do I set them up?" -- Two question marks trigger the multi-part rule, and "how do" matches the keyword list.

**Why this boundary?** The goal was to be conservative. Sending a simple query to the 70B model wastes tokens but still gives a correct answer. Sending a complex query to the 8B model risks a bad answer. So the router is intentionally biased toward the complex classification to protect answer quality.

**Misclassification example:** The query "how do I reset my password" gets classified as complex because of the "how do" keyword trigger. This is a simple factual lookup that the 8B model could handle fine. The keyword matching is too broad -- it cannot distinguish between "how do I do X" (simple instruction) and "how does the billing engine calculate prorated charges" (genuinely complex).

**Improvement without an LLM:** I would add a second-pass filter that checks the full phrase context, not just keyword presence. For example, "how do I" followed by a short sentence (under 8 words) would be reclassified back to simple. I would also add a curated whitelist of known simple question patterns like password resets, login issues, and account setup, so they bypass the keyword rules entirely.

---

## Q2 -- Retrieval Failures

**Query:** "What happens if I downgrade my plan mid-cycle?"

**What the system retrieved:** The BM25 pipeline returned chunks from the pricing sheet that describe the features of each plan tier (Pro, Enterprise, etc.) but nothing about downgrade policies or pro-rated billing. The top chunks matched on "plan" and "downgrade" as individual tokens, but the actual downgrade policy was buried inside a terms-of-service PDF where the word "downgrade" appeared only once in a longer paragraph alongside legal language.

**Why it failed:** BM25 is a lexical matching algorithm. It scores documents based on exact token overlap. The terms-of-service chunk had low term frequency for "downgrade" relative to its total length, so its BM25 score was lower than the pricing sheet chunks where "plan" appeared densely. BM25 has no understanding of meaning -- it cannot know that "downgrade mid-cycle" is semantically closer to "cancellation and refund policy" than to "plan feature comparison."

**What would fix it:** The most direct fix would be a hybrid retrieval approach -- combine BM25 with a dense vector search using sentence embeddings (e.g., using a model like `all-MiniLM-L6-v2`). The vector search would catch semantic similarity even when the exact keywords do not match. The two score lists can be merged using reciprocal rank fusion. This would preserve BM25's strength with exact keyword matches while covering the semantic gaps.

---

## Q3 -- Cost and Scale

**Assumptions:**
- 5,000 queries per day.
- The router splits roughly 60% simple, 40% complex (based on keyword trigger frequency in typical customer support questions).
- Average input tokens per query: ~1,200 (system prompt + 5 retrieved chunks of ~350 words each + conversation history).
- Average output tokens per query: ~250 (concise support answer, max_tokens is capped at 1,024).

**Daily token breakdown:**

| Model | Queries/day | Input tokens | Output tokens | Total tokens |
|---|---|---|---|---|
| llama-3.1-8b-instant | 3,000 | 3,600,000 | 750,000 | 4,350,000 |
| llama-3.3-70b-versatile | 2,000 | 2,400,000 | 500,000 | 2,900,000 |
| **Total** | **5,000** | **6,000,000** | **1,250,000** | **7,250,000** |

**Biggest cost driver:** Input tokens on the 70B model. The 70B model handles 40% of queries but its per-token cost is significantly higher. The 5 retrieved chunks dominate the input token count regardless of query complexity.

**Highest-ROI change:** Reduce the number of retrieved chunks from 5 to 3 for simple queries. Simple queries are typically answered by a single chunk, so the extra 2 chunks add ~500 input tokens per request with minimal benefit. For 3,000 simple queries per day, that saves roughly 1,500,000 input tokens daily, cutting total input costs by 25%.

**Optimisation I would avoid:** Caching LLM responses for repeated queries. Customer support queries are rarely identical in phrasing, and even small wording differences would be cache misses. A stale cached answer could also be wrong if the documentation has been updated. The cache hit rate would be too low to justify the added complexity.

---

## Q4 -- What Is Broken

**The most significant flaw:** The conversation memory is stored in-memory in a Python dictionary (`_conversations`). On Vercel's serverless deployment, every function invocation can spin up a new instance. This means the conversation dictionary is empty on each cold start, so multi-turn conversations are effectively broken in production. A user asks a follow-up question, the server has no record of the previous exchange, and the LLM answers without any conversational context.

**Why I shipped with it anyway:** The in-memory approach was the fastest way to demonstrate that the conversation feature works functionally. It passes the requirement during a local demo or testing session where the server stays alive. Adding a persistent store (Redis, a database) would have introduced infrastructure dependencies, environment configuration, and additional deployment complexity that was outside the scope of a take-home assignment focused on the RAG pipeline and routing logic.

**The fix:** Swap the dictionary for a lightweight external store. The simplest option would be Redis with a TTL of 30 minutes per conversation. Each conversation ID maps to a JSON list of messages. The code change is minimal -- replace `_conversations.get(conv_id, [])` with a Redis GET, and replace the append with a Redis SET. This would survive cold starts and scale horizontally across multiple serverless instances.

---

## AI Usage

First did a research on how to build a RAG system without using any external things and found BM25 can be used.

I utilized AI coding assistants extensively to accelerate the boilerplate generation and implement the architecture. Rather than relying on the AI to invent the architecture, I used a "Project Brief" prompting strategy, where I explicitly constrained the AI's output to strictly adhere to the assignment rules (e.g., forcing pure-math BM25 instead of vector embeddings, and strictly enforcing the API contract).

Below are the exact, foundational mega-prompts I used to generate the core layers of the application.

### Prompt 1: Backend Architecture & BM25 Pipeline
*Note: After providing this prompt, I used short iterative conversational commands (e.g., "Extract the text using PyMuPDF", "Save the index as a pickle file") to refine the generated code.*

**The Prompt:**
> **Project Overview**
> We are building a customer support chatbot API for a fictional project management SaaS company called ClearPath. The chatbot answers user questions by retrieving relevant content from 30 provided PDF documents and generating a response using an LLM.
>
> **Architecture Requirements & Strict Constraints**
> 1. Layer 1: RAG Pipeline. Do NOT use external RAG libraries or managed retrieval services. We must build the chunking, embedding, and retrieval logic from scratch. You MUST use BM25 (rank_bm25) for lexical search to avoid using external embedding models.
> 2. Layer 2: Model Router. This router MUST be a deterministic, rule-based classifier. Do NOT use an LLM call to make the routing decision. Route "simple" to llama-3.1-8b-instant and "complex" to llama-3.3-70b-versatile. Log to a JSONL file.
> 3. Layer 3: Output Evaluator. Flag "no_context", "refusals", and "internal_data_leak" (domain-specific check).
>
> Generate the Python FastAPI backend adhering exactly to these constraints.

### Prompt 2: Frontend UI & Debug Panel Integration
*Note: I provided my existing React chat UI component and asked the AI to modify it to fit the ClearPath requirements.*

**The Prompt:**
> I have an existing React chat interface. We need to repurpose it for a customer support chatbot and connect it to a local FastAPI backend running on `http://127.0.0.1:8000`.
>
> **Constraints:**
> 1. Preserve my existing Tailwind styling.
> 2. Build a Debug Panel that displays the LLM routing and token usage data from the backend metadata object.
> 3. If the backend flags a response (evaluator_flags > 0), the UI must render a visible warning label: "Warning -- Low confidence -- please verify with support."
>
> Update my state management to hit the `/query` endpoint and append the debug panel to the layout.
