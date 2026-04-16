## TokenSmith Query Decomposition (CS 6423)

This repo tracks my course project: **Query Decomposition and Planning for Multi-Hop Retrieval in TokenSmith**.

**Core idea:** TokenSmith’s baseline answers each question with a single retrieval pass (top-\(k\) chunks) followed by one generation step. This works for simple, single-hop questions but degrades on **multi-hop / comparative / multi-part** questions. I’m adding a DB-inspired “query planner” layer:

- **Query classifier**: route `simple` questions to baseline (no extra overhead).
- **Query decomposer / planner**: for `complex` questions, generate 2–4 sub-queries (later: dependency graph for multi-hop).
- **Evidence merge + synthesis**: retrieve per sub-query, dedupe/merge evidence, then synthesize a final answer with citations.

### Important checkpoint constraints

- **Corpus**: use **only** `silberschatz.pdf` for this project.
- **Frozen baseline**: ensure `top_k = 10` in TokenSmith’s `config/config.yaml` and **do not change anything else** in that config.
- **Frontend must work**: TokenSmith-Frontend must be able to hit the backend API and show citation-backed answers.

This repo is intended to be the **public** code/repro link placed at the top of the checkpoint progress report.

---

## Repro (local)

### Prereqs
- Python + conda/miniconda (TokenSmith manages deps in a conda env called `tokensmith`)
- Node 18+ / npm (for TokenSmith-Frontend)
- Local GGUF models placed where TokenSmith expects them (see `TokenSmith/config/config.yaml`)

### 1) TokenSmith: extract + index Silberschatz

From the workspace root (the folder containing both repos):

```bash
cd TokenSmith

# follow TokenSmith README steps 1–6
make build
conda activate tokensmith

# place silberschatz.pdf at TokenSmith/data/chapters/silberschatz.pdf
make run-extract

# (optional) ensure TokenSmith/data/ contains only the Silberschatz extracted markdown
# because current index mode uses the first data/*.md it finds.

make run-index
```

Expected artifacts (after indexing):
- `TokenSmith/index/sections/textbook_index.faiss`
- `TokenSmith/index/sections/textbook_index_bm25.pkl`
- `TokenSmith/index/sections/textbook_index_chunks.pkl`
- `TokenSmith/index/sections/textbook_index_sources.pkl`
- `TokenSmith/index/sections/textbook_index_meta.pkl`
- `TokenSmith/index/sections/textbook_index_page_to_chunk_map.json`

### 2) TokenSmith backend API (for frontend)

```bash
cd TokenSmith
conda activate tokensmith
uvicorn src.api_server:app --reload --port 8000
```

Sanity check:
- `http://localhost:8000/api/health`

### 3) TokenSmith-Frontend

```bash
cd TokenSmith-Frontend
npm install
npm run dev
```

Open:
- `http://localhost:5173`

The frontend expects the backend at `http://localhost:8000` and uses:
- `POST /api/chat`
- `POST /api/chat/stream`

---

## Benchmark questions

Starter benchmark questions live in:
- `benchmark/questions.json`

It contains:
- One **connected learning episode** (3–5 questions) to copy into the checkpoint appendix.
- Additional simple/complex questions to expand into a larger evaluation set (30+).

For checkpoint submission, the appendix requires **for each question**:
- the question
- the best 1–2 **ground-truth textbook chunks** (paste from the UI)

---

## Planned implementation (to be added)

Target structure (in this repo):
- `planner/`: classifier, decomposer, evidence merger, synthesizer, and an orchestrating pipeline
- `benchmark/`: question set + evaluation scripts + results
- `reports/`: checkpoint/progress/final report sources

Integration points in TokenSmith:
- CLI pipeline: `TokenSmith/src/main.py` (`get_answer(...)`)
- API pipeline: `TokenSmith/src/api_server.py` (retrieval + generation per request)

