# AI Engineering Project Plan (21 Weeks)

One project per week, one repo that grows over time. Default stack is **Python + Ollama** (Qwen 2.5 7B or Llama 3.1 8B) unless noted. Use the companion checklist (`ai-engineering-checklist.md`) to track topics.

---

## Phase 1: Foundations (Weeks 1–3)

### Week 1: Local Model Bench + Tokenizer Side-Quest ✅ Done
- **What:** CLI that benchmarks 3–4 local models across quantization levels (Q4, Q5, Q8) for speed, memory, and quality. Side-quest: train a small BPE tokenizer on Tamil + English text and compare token counts.
- **Stack:** Ollama, llama.cpp, Typer, Pandas, Rich
- **Learn:** Model serving basics, quantization, context windows, sampling parameters, tokenization
- **Study:** karpathy/minbpe; rasbt/LLMs-from-scratch ch. 2

### Week 2: Structured Extractor ✅ Done
- **What:** Messy text in, validated JSON out, with retries when output breaks the schema. Add streaming responses and async calls.
- **Stack:** Pydantic, Instructor or raw JSON mode, FastAPI (optional: Pydantic AI)
- **Learn:** Prompt and context engineering, structured outputs, validation, streaming
- **Study:** Hugging Face LLM Course ch. 1–2

### Week 3: Tool-Calling Agent from Scratch ✅ Done
- **What:** Hand-written agent loop with 3–4 tools. No frameworks.
- **Stack:** Python, httpx, Ollama tool calling
- **Learn:** How agents work under the hood; limits of small models
- **Study:** rasbt/LLMs-from-scratch ch. 3–4 (attention, GPT architecture)

---

## Phase 2: Retrieval (Weeks 4–6)

### Week 4: Docs Q&A (Naive RAG)
- **What:** Chat with your own documents: chunk, embed, retrieve, answer with citations.
- **Stack:** nomic-embed-text, pgvector (Docker), PyMuPDF, FastAPI, Streamlit
- **Study:** LLM Zoomcamp: RAG modules

### Week 5: RAG Eval Harness
- **What:** 50 golden Q&A pairs; score retrieval recall, faithfulness, hallucination rate.
- **Stack:** Pytest, Ragas or DeepEval, LLM-as-judge
- **Study:** LLM Zoomcamp: evaluation module
- **Optional:** DSPy for programmatic prompt optimization

### Week 6: Advanced + Self-Correcting RAG + Frameworks
- **What:** Hybrid search, reranker, query rewriting, self-grading retrieval. Rebuild Week 4 in **LangChain** and **LlamaIndex**; compare on evals.
- **Stack:** rank_bm25, pgvector, bge-reranker, LangChain, LlamaIndex, LangGraph
- **Optional:** GraphRAG / knowledge-graph retrieval
- **Study:** Sumanth077/Hands-On-AI-Engineering

---

## Phase 3: Production Concerns (Weeks 7–9)

### Week 7: Observability, Guardrails + Red Team
- **What:** Tracing, cost tracking, PII masking. Attack your RAG with poisoned docs and injections, then defend it.
- **Stack:** Langfuse (Docker), Presidio, Llama Guard or NeMo Guardrails
- **Study:** W&B AI Academy; OWASP Top 10 for LLM Applications

### Week 8: LLM Event Pipeline + Cost Controls
- **What:** Queue-driven workers enriching events with an LLM (batching, retries, DLQ, idempotency). Add a model gateway with routing (cheap model first, escalate if needed), prompt/semantic caching, and fallbacks.
- **Stack:** LocalStack (SQS/Lambda) or Redis Streams, Python or .NET 8, LiteLLM, Redis cache (optional: Semantic Kernel / MS.Extensions.AI)
- **Learn:** LLMs in distributed systems; cost and latency optimization

### Week 9: Serving at Scale
- **What:** Serve a model with vLLM; load-test it. Compare throughput vs. Ollama, measure effects of batching, KV cache, and quantization formats (GGUF, AWQ, GPTQ). Compare against a managed option (Amazon Bedrock).
- **Stack:** vLLM (Colab/cloud GPU if needed), Locust or k6, Amazon Bedrock
- **Learn:** Inference internals, throughput vs. latency, self-host vs. managed tradeoffs

---

## Phase 4: Agents & Integration (Weeks 10–14)

### Week 10: MCP Server + Text-to-SQL
- **What:** Expose your RAG index and a SQL database as MCP tools, including a text-to-SQL tool with safe, read-only execution.
- **Stack:** MCP Python SDK (FastMCP), SQLite/Postgres, MCP Inspector, Claude Desktop; try **smolagents** as a client
- **Study:** Hugging Face MCP Course

### Week 11: Mini Coding Agent (LangGraph)
- **What:** Reads files, edits code, runs tests, loops until green. LangGraph state machine with checkpointing and human-in-the-loop. Add agent memory (short-term summaries, long-term store) and trajectory evals.
- **Stack:** LangGraph, subprocess sandbox, Qwen2.5-Coder
- **Study:** Hugging Face Agents Course

### Week 12: PR Review Bot
- **What:** GitHub webhook that reviews diffs with repo context and posts inline comments.
- **Stack:** FastAPI, GitHub App + PyGithub, tree-sitter, smee.io
- **Study:** patchy631/ai-engineering-hub

### Week 13: Research Agent (Advanced LangGraph)
- **What:** Plans, searches, reads, writes cited reports under a step budget. Subgraphs, parallel fan-out/merge, resumable runs. Try a reasoning model and compare.
- **Stack:** LangGraph, SearXNG, trafilatura
- **Study:** Hugging Face Context Course

### Week 14: LLM Council (Multi-Agent)
- **What:** Models answer, critique each other, a chair synthesizes. Build plain, then with **CrewAI or AutoGen**; compare on evals.
- **Stack:** Ollama (multiple models), asyncio, CrewAI or AutoGen
- **Study:** karpathy/llm-council

---

## Phase 5: Multimodal (Week 15)

### Week 15: Vision + Document Understanding
- **What:** Parse complex PDFs (tables, scanned pages, charts) with a vision-language model and layout-aware parsers; feed results into your RAG. Compare against text-only extraction.
- **Stack:** Qwen2.5-VL or Llama 3.2 Vision (Ollama), Docling or Marker, your RAG + evals
- **Learn:** Multimodal models, OCR vs. VLM tradeoffs, multimodal RAG

---

## Phase 6: Fine-Tuning (Weeks 16–18)

### Week 16: Supervised Fine-Tuning (LoRA/QLoRA)
- **What:** Build a 500–1,000 example dataset (hand-labelled + synthetic, filtered) for the Week 2 task. QLoRA fine-tune a 3B–7B model, merge, export to GGUF, serve in Ollama. Compare fine-tuned 3B vs. prompted 7B/14B.
- **Stack:** Unsloth, Transformers + PEFT + TRL, llama.cpp, Colab/Kaggle GPUs
- **Study:** rasbt/LLMs-from-scratch ch. 6–7; Hugging Face LLM Course fine-tuning chapters; Unsloth notebooks

### Week 17: Fine-Tune for Retrieval
- **What:** Fine-tune an embedding model (and optionally a reranker) on question/passage pairs from your docs. Measure recall before and after.
- **Stack:** sentence-transformers, Week 5 eval harness

### Week 18: Preference Tuning (DPO) + Distillation
- **What:** Build a chosen/rejected dataset and run DPO to change behaviour. Distil a large model's outputs into a small model for a narrow skill.
- **Stack:** TRL (DPOTrainer), Unsloth

---

## Phase 7: Voice + Capstone (Weeks 19–21)

### Week 19: Voice Pipeline
- **What:** Speech in, LLM reasons, speech out, streaming with per-stage latency numbers.
- **Stack:** Silero VAD, faster-whisper, Ollama, Piper, WebSockets

### Weeks 20–21: Capstone: Personal Second Brain
- **What:** RAG + MCP + memory + evals + tracing over your notes and docs, deployed and used daily. Write an architecture doc and a short demo video.
- **Stack:** Docker, AWS (ECS or Lambda, RDS pgvector, optional Bedrock), GitHub Actions running evals in CI
- **Study:** Berkeley Advanced LLM Agents (optional)

---

## Ongoing (Alongside Every Week)
- **LLM system design practice:** one case study per week (e.g., design a support chatbot, a RAG platform for 10M docs, a code assistant). Tie each to the project you just built.
- **Write-ups:** one short blog/LinkedIn post per phase with real numbers.
- **Reading:** Chip Huyen's *AI Engineering* book as a backbone reference.

---

## Framework Coverage

| Framework | Where |
|---|---|
| LangChain | Week 6 |
| LlamaIndex | Week 6 |
| LangGraph | Weeks 6, 11, 13 |
| smolagents | Week 10 |
| CrewAI / AutoGen | Week 14 |
| LiteLLM | Week 8 |
| vLLM | Week 9 |
| Unsloth / TRL / PEFT | Weeks 16–18 |
| Pydantic AI (optional) | Week 2 |
| DSPy (optional) | After Week 5 |
| Semantic Kernel / MS.Extensions.AI (optional) | Week 8 |

---

## Resources

**Internals**
- [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- [karpathy/minbpe](https://github.com/karpathy/minbpe)
- [karpathy/nanochat](https://github.com/karpathy/nanochat) (read, don't train; assumes NVIDIA GPUs)

**Courses (free)**
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1)
- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/en/unit0/introduction) (+ MCP and Context courses)
- [LLM Zoomcamp (DataTalks.Club)](https://datatalks.club/blog/llm-zoomcamp.html)
- [W&B AI Academy](https://wandb.ai/site/courses/)
- [Berkeley Advanced LLM Agents](https://rdi.berkeley.edu/adv-llm-agents/sp25)

**Project idea banks**
- [patchy631/ai-engineering-hub](https://github.com/patchy631/ai-engineering-hub)
- [Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps)
- [Sumanth077/Hands-On-AI-Engineering](https://github.com/Sumanth077/Hands-On-AI-Engineering)
- [karpathy/llm-council](https://github.com/karpathy/llm-council)

---

## Rules

1. **One repo**, one folder per week, each with a README: what you built, what broke, the numbers.
2. **No frameworks until Week 6.** Build by hand first.
3. **Evals from Week 5 onward.** Measure every later project.
4. **Allow slippage.** Build in 2–3 buffer weeks.
5. **Check hardware in Week 1.** Plan Colab or a cloud GPU for Weeks 9, 13, 16–18, and 19 if a 14B model doesn't run well locally.

---

## Progress Tracker

| Week | Project | Status |
|---|---|---|
| 1 | Local Model Bench + Tokenizer Side-Quest | ✅ Done |
| 2 | Structured Extractor | ☐ |
| 3 | Tool-Calling Agent from Scratch | ☐ |
| 4 | Docs Q&A (Naive RAG) | ☐ |
| 5 | RAG Eval Harness | ☐ |
| 6 | Advanced + Self-Correcting RAG + Frameworks | ☐ |
| 7 | Observability, Guardrails + Red Team | ☐ |
| 8 | LLM Event Pipeline + Cost Controls | ☐ |
| 9 | Serving at Scale | ☐ |
| 10 | MCP Server + Text-to-SQL | ☐ |
| 11 | Mini Coding Agent (LangGraph) | ☐ |
| 12 | PR Review Bot | ☐ |
| 13 | Research Agent (Advanced LangGraph) | ☐ |
| 14 | LLM Council (Multi-Agent) | ☐ |
| 15 | Vision + Document Understanding | ☐ |
| 16 | Supervised Fine-Tuning (LoRA/QLoRA) | ☐ |
| 17 | Fine-Tune for Retrieval | ☐ |
| 18 | Preference Tuning (DPO) + Distillation | ☐ |
| 19 | Voice Pipeline | ☐ |
| 20–21 | Capstone: Personal Second Brain | ☐ |
