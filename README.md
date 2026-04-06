# RAG_PROJECT — ask questions over your PDFs

This project is a small **question-and-answer** app that reads two PDFs, splits them into searchable pieces, and answers your questions **using only text it found** in those documents. It is not a general chatbot: if the answer is not in the PDFs, it should say so or find the least relavent one.

This README explains **what I built**, **why I chose it** (including trade-offs), and **how to run everything** step by step.

---

## What you need on your computer

- **Python 3.10+** (3.11 or 3.12 is fine).
- Enough disk space for **PyTorch** and the **embedding models** (a few gigabytes the first time they download).
- **Internet** the first time you build the index (models download from Hugging Face).
- An **OpenAI API key** (optional). Without it, the app still works: it shows a short **quote** from the best matching passage instead of a full written answer.

---

## What files the project expects

Put these two PDFs in the folder **`data/raw/`** (create the folders if they are missing):

| File name | What it is |
|-----------|------------|
| `Attention_is_all_you_need.pdf` | The Transformer paper (example technical doc). |
| `EU_AI_Act_Doc.pdf` | A **summary** of the EU AI Act (not legal advice). |

If the PDFs live somewhere else on your machine, you can copy them in with the setup script (you can refer to Step 4 below).

---
## Findings and trade-offs (design decisions)

### Retrieval: dense + keywords + fusion

- **Dense search (embeddings + FAISS)** finds passages that are **similar in meaning** even when wording differs.  
  **Trade-off:** Short or unusual phrases can rank oddly; similar-looking chunks from the “wrong” document can score high.

- **BM25 (keyword search)** is strong when the user types **exact words** from the document.  
  **Trade-off:** It is weak on paraphrases (“authorize” vs “permit”).

- **Hybrid retrieval** combines both and uses **RRF (reciprocal rank fusion)** so we do not have to force two different score types onto one scale.  
  **Trade-off:** More moving parts and more CPU than dense-only.

- **Cross-encoder reranking** scores each **(query, chunk)** pair more accurately than cosine alone.  
  **Trade-off:** Slower first query (model load) and more compute per request.

- **Lexical phrase priority** scans the corpus for queries where **tokens appear in order** in a short window (e.g. “track, document”).  
  **Trade-off:** Extra pass over chunks; tuned rules to avoid junk matches on very short queries.

**Finding:** For a **two-document** corpus, this stack noticeably improved **subsection** and **short-query** behaviour compared to dense-only + large chunks.

### Chunking and PDF quirks

- We use **paragraph-first** splitting, then **sentence** splitting for very long blocks, with **overlap** so facts on chunk edges are not lost.  
  **Trade-off:** Smaller chunks mean **more** chunks → longer indexing and slightly more retrieval noise if `top_k` is too low.

- The EU Act file is a **summary PDF**, not perfect law text. We added **targeted preprocessing** (heading and bullet splits) so list-style sections do not end up in one giant blob.  
  **Trade-off:** Rules are **specific to this PDF style**; another document might need different rules.

- **Chunk IDs include the page number** so IDs are unique across the whole book. Earlier, per-page indices collided and broke lookups—this was a real bug, not a cosmetic fix.

### Embeddings and index

- Default model is **MPNet-class** (see `config.py` / `.env.example`), not the smallest MiniLM.  
  **Trade-off:** Better quality and **768-dimensional** vectors vs faster downloads and smaller models.

- FAISS uses **exact** inner product search (`IndexFlatIP`) on **normalized** vectors (cosine-like ranking).  
  **Trade-off:** Fine for tens–hundreds of chunks; for **millions** of chunks you would switch to an approximate index (IVF, HNSW, etc.).

### Generation and output

- Answers are supposed to **stick to retrieved context**; the system prompt asks for citations and to refuse when the text is not there.  
  **Trade-off:** The model can still slip; evaluation uses **lightweight heuristics**, not a human gold standard.

- **API and log output** flattens newlines in answers and previews for readable JSON.  
  **Trade-off:** The **LLM still sees** original newlines inside chunk text when generating.

### Evaluation

- **`test_queries.json`** checks whether the **right PDF** appears in the top results and whether **hint words** show up—not formal legal correctness.  
  **Trade-off:** Cheap to run, easy to explain, but **not** a substitute for domain expert review.

### Packaging and tooling

- **`pip install -e .`** (via `requirements.txt`) makes imports reliable in **IDEs, scripts, and pytest**.  
  **Trade-off:** You must run `pip install -r requirements.txt` from the **project root** so `-e .` resolves correctly.


---

# How to run the project (full walkthrough)

Follow these steps **in order** from the **project root** (the folder that contains `README.md`).

### Step 1 — Open a terminal in the project folder

On Windows (PowerShell), for example:

```powershell
cd D:\yash\projects\CeDAR_Project
```
(Use your actual path if it is different.)

### Step 2 — Create and activate a virtual environment

This keeps CeDAR’s libraries separate from other Python projects.

```powershell
python -m venv .venv

#It is not neccesary to run the below command if you are using terminal in any particular IDE(Visual Studio) as you can easly run the files aas local user.
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install dependencies and the local `app` package

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The file `requirements.txt` includes **`-e .`**, which installs **this repo** as a package so `import app` works from anywhere. That avoids “No module named app” errors.

### Step 4 — (Optional) Copy the PDFs into `data/raw/`

If you already placed the two PDFs manually under `data/raw/`, skip this.

Otherwise, copy from paths you have:

```powershell
python scripts\setup_data.py  
#It tries to check the files, are the files present and are they present at particular location  aswell.
--attention-src "C:\path\to\Attention_is_all_you_need.pdf" --eu-act-src "C:\path\to\EU_AI_Act_Doc.pdf"
```

You can also set environment variables `SOURCE_ATTENTION_PDF` and `SOURCE_EU_AI_ACT_PDF` instead of using the command-line flags.

### Step 5 — Configure environment variables (optional but recommended)

```powershell
copy .env.example .env
```

Then edit **`.env`**:

- Set **`OPENAI_API_KEY=`** if you want **full answers** from the language model.
- Leave it empty if you only want **retrieval + a short excerpt** (no bill from OpenAI for chat).

Other settings (chunk size, hybrid search, etc.) are documented in **`.env.example`**. After you change chunk size or embedding model, you **must** run Step 6 again.

### Step 6 — Build the search index (required before asking questions)

This reads the PDFs, cleans text, splits it into **chunks**, computes **embeddings**, and saves a **FAISS** index under `data/processed/`.

```powershell
python scripts\build_index.py
```

**Run this again** whenever you:

- Replace or update the PDFs,
- Change chunk size or overlap in `.env`,
- Change the embedding model name.

First run can take **several minutes** while models download.

### Step 7 — Try a question from the command line

```powershell
python scripts\run_query.py "General purpose AI (GPAI):" 
```

You should see JSON with an **answer** (or excerpt) and **sources**. A line is also appended to **`outputs/query_logs.jsonl`**.

### Step 8 — Run the tests (optional)

```powershell
python -m pytest tests -m "not integration" -q
```

The **integration** test is slower (builds a tiny index); run it with:

```powershell
python -m pytest tests -m integration -q
```

### Step 9 — Run evaluation (optional)

Uses **`data/eval/test_queries.json`** and writes **`outputs/eval_results.json`**.

```powershell
python scripts\evaluate.py
```

### Step 10 — Start the web UI

```powershell
python scripts\run_server.py --reload
```

The terminal will print a URL like **`http://127.0.0.1:8000/`**.  
**Ctrl+click** that line in many terminals, or **copy and paste** it into Chrome or Edge.

- **`--reload`** restarts the server when you edit code (handy for development).
- API documentation (try requests in the browser) is at **`http://127.0.0.1:8000/docs`**.

---

## Quick command reference

| What you want | Command |
|---------------|---------|
| Install everything | `pip install -r requirements.txt` |
| Put PDFs in place | `python scripts/setup_data.py ...` |
| Build / rebuild index | `python scripts/build_index.py` |
| One question (CLI) | `python scripts/run_query.py "your question"` |
| Web + API | `python scripts/run_server.py --reload` |
| Tests | `python -m pytest tests -m "not integration" -q` |
| Evaluation | `python scripts/evaluate.py` |
| Interview notes → Word | `python scripts/build_interview_docx.py` |

---

## Docker (optional)

```bash
docker build -f deploy/Dockerfile -t rag-doc-qa .
```

More notes: **`deploy/cloud_notes.md`**.

---

## Deeper interview prep

- **`docs/PROJECT_WALKTHROUGH_INTERVIEW.md`** — structure, glossary, common follow-up questions.  
- Regenerate Word: **`python scripts/build_interview_docx.py`**

---

## Legal note

The PDFs are **not** my originals. The EU AI Act file is a **summary** for demonstration only — **not legal advice**.
