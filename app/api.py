from __future__ import annotations

"""
HTTP front door: JSON API for queries and a single-page HTML demo.

Business logic stays in pipeline.answer_query — this file only validates input and renders UI.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app import config
from app.pipeline import answer_query

app = FastAPI(title="rag-qa", version="1.0.0")


class QueryBody(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    # Self-contained HTML+CSS+JS; POSTs to /query and escapes model output in JS for safety.
    return """<!DOCTYPE html>
<!-- CeDAR UI: static shell; all answers from POST /query (see script at bottom). -->
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>CeDAR — Ask your documents</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
  <link href="https://fonts.googleapis.com/css2?family=Figtree:ital,wght@0,400;0,600;0,700;1,400&family=Fraunces:ital,opsz,wght@0,9..144,600;0,9..144,700;1,9..144,600&display=swap" rel="stylesheet"/>
  <style>
    :root {
      --bg0: #0c0e12;
      --bg1: #12151c;
      --card: #181c26;
      --card2: #1e2430;
      --border: rgba(255,255,255,.08);
      --text: #e8eaef;
      --muted: #8b93a7;
      --accent: #5eead4;
      --accent-dim: rgba(94, 234, 212, .15);
      --warn: #fcd34d;
      --glow: rgba(94, 234, 212, .35);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Figtree", system-ui, sans-serif;
      color: var(--text);
      background: var(--bg0);
      background-image:
        radial-gradient(ellipse 120% 80% at 50% -20%, rgba(94, 234, 212, .12), transparent 50%),
        radial-gradient(ellipse 60% 40% at 100% 50%, rgba(99, 102, 241, .08), transparent 45%),
        linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 100%);
    }
    .wrap { max-width: 820px; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }
    header {
      text-align: center;
      margin-bottom: 2.5rem;
      animation: fadeUp .7s ease-out both;
    }
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .badge {
      display: inline-block;
      font-size: .72rem;
      font-weight: 600;
      letter-spacing: .14em;
      text-transform: uppercase;
      color: var(--accent);
      background: var(--accent-dim);
      border: 1px solid rgba(94, 234, 212, .25);
      padding: .35rem .75rem;
      border-radius: 999px;
      margin-bottom: 1rem;
    }
    h1 {
      font-family: "Fraunces", Georgia, serif;
      font-size: clamp(1.85rem, 5vw, 2.45rem);
      font-weight: 700;
      line-height: 1.15;
      margin: 0 0 .6rem;
      background: linear-gradient(135deg, #fff 0%, #a5f3e0 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .sub { color: var(--muted); font-size: 1.05rem; max-width: 32rem; margin: 0 auto; line-height: 1.5; }
    .panel {
      background: linear-gradient(145deg, var(--card) 0%, var(--card2) 100%);
      border: 1px solid var(--border);
      border-radius: 1.25rem;
      padding: 1.5rem 1.35rem;
      box-shadow: 0 0 0 1px rgba(255,255,255,.03) inset, 0 24px 48px -24px rgba(0,0,0,.5);
      animation: fadeUp .75s ease-out .1s both;
    }
    label { display: block; font-weight: 600; font-size: .9rem; color: var(--muted); margin-bottom: .5rem; }
    textarea {
      width: 100%;
      min-height: 110px;
      resize: vertical;
      padding: 1rem 1.1rem;
      font: inherit;
      font-size: 1rem;
      line-height: 1.5;
      color: var(--text);
      background: rgba(0,0,0,.25);
      border: 1px solid var(--border);
      border-radius: .75rem;
      transition: border-color .2s, box-shadow .2s;
    }
    textarea:focus {
      outline: none;
      border-color: rgba(94, 234, 212, .4);
      box-shadow: 0 0 0 3px var(--glow);
    }
    textarea::placeholder { color: #5c6578; }
    .row { display: flex; flex-wrap: wrap; gap: .75rem; align-items: center; margin-top: 1.1rem; }
    button[type="submit"] {
      font: inherit;
      font-weight: 700;
      font-size: .95rem;
      cursor: pointer;
      border: none;
      border-radius: .65rem;
      padding: .75rem 1.6rem;
      color: #042f2e;
      background: linear-gradient(135deg, #5eead4 0%, #2dd4bf 100%);
      box-shadow: 0 4px 20px -4px var(--glow);
      transition: transform .15s, box-shadow .2s, filter .2s;
    }
    button[type="submit"]:hover:not(:disabled) {
      transform: translateY(-1px);
      filter: brightness(1.05);
      box-shadow: 0 8px 28px -6px var(--glow);
    }
    button[type="submit"]:active:not(:disabled) { transform: translateY(0); }
    button[type="submit"]:disabled {
      opacity: .65;
      cursor: wait;
      transform: none;
    }
    .hint { font-size: .8rem; color: var(--muted); }
    .hint code { font-size: .78rem; background: rgba(0,0,0,.3); padding: .15rem .4rem; border-radius: .35rem; }
    #out { margin-top: 2rem; animation: fadeUp .5s ease-out both; }
    #out:empty { margin-top: 0; animation: none; }
    .section-title {
      font-family: "Fraunces", Georgia, serif;
      font-size: 1.15rem;
      font-weight: 600;
      margin: 0 0 1rem;
      color: var(--warn);
      display: flex;
      align-items: center;
      gap: .5rem;
    }
    .section-title::before {
      content: "";
      width: 4px;
      height: 1.1em;
      background: linear-gradient(180deg, var(--accent), transparent);
      border-radius: 2px;
    }
    .answer {
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.65;
      padding: 1.15rem 1.25rem;
      background: rgba(0,0,0,.22);
      border: 1px solid var(--border);
      border-radius: .85rem;
      margin-bottom: 1.75rem;
    }
    .sources { display: flex; flex-direction: column; gap: .85rem; }
    .src {
      padding: 1rem 1.1rem;
      border-radius: .75rem;
      border: 1px solid var(--border);
      background: rgba(0,0,0,.18);
      transition: border-color .2s, background .2s;
    }
    .src:hover { border-color: rgba(94, 234, 212, .2); background: rgba(94, 234, 212, .04); }
    .src-top { display: flex; flex-wrap: wrap; align-items: baseline; gap: .5rem .75rem; margin-bottom: .5rem; }
    .src-name { font-weight: 700; font-size: .92rem; }
    .src-page { color: var(--muted); font-size: .85rem; }
    .score-pill {
      font-size: .72rem;
      font-weight: 600;
      color: var(--accent);
      background: var(--accent-dim);
      padding: .2rem .5rem;
      border-radius: 999px;
      margin-left: auto;
    }
    .src-preview { font-size: .88rem; color: #b4bac8; line-height: 1.5; }
    .src-head { font-style: italic; color: var(--muted); font-size: .85rem; margin-bottom: .35rem; }
    .err {
      padding: 1rem;
      border-radius: .75rem;
      background: rgba(239, 68, 68, .12);
      border: 1px solid rgba(239, 68, 68, .3);
      color: #fecaca;
      font-size: .9rem;
      white-space: pre-wrap;
    }
    footer {
      text-align: center;
      margin-top: 3rem;
      font-size: .78rem;
      color: #5c6578;
      animation: fadeUp .8s ease-out .2s both;
    }
    footer a { color: var(--muted); }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <span class="badge">Retrieval-augmented Q&amp;A</span>
      <h1>Ask your documents</h1>
      <p class="sub">Questions run against your indexed PDFs. Answers use only retrieved passages—grounded, not guessed.</p>
    </header>
    <div class="panel">
      <form id="f">
        <label for="q">Your question</label>
        <textarea id="q" name="query" placeholder="e.g. What obligations apply to GPAI model providers?"></textarea>
        <div class="row">
          <button type="submit" id="btn">Run retrieval</button>
          <span class="hint">API: <code>POST /query</code> with <code>{"query":"…"}</code></span>
        </div>
      </form>
    </div>
    <div id="out"></div>
    <footer>Powered by CeDAR · FAISS + hybrid search · <a href="/docs">OpenAPI docs</a></footer>
  </div>
  <script>
  // POST /query, then build DOM strings with esc() so model text can’t run as HTML/JS.
  function esc(s) {
    if (s == null) return '';
    const d = document.createElement('div');
    d.textContent = String(s);
    return d.innerHTML;
  }
  document.getElementById('f').onsubmit = async (e) => {
    e.preventDefault();
    const qel = document.getElementById('q');
    const query = qel.value.trim();
    if (!query) return;
    const btn = document.getElementById('btn');
    const out = document.getElementById('out');
    btn.disabled = true;
    const prev = btn.textContent;
    btn.textContent = 'Retrieving…';
    out.innerHTML = '';
    try {
      const r = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });
      const j = await r.json();
      if (!r.ok) {
        out.innerHTML = '<div class="err">' + esc(JSON.stringify(j, null, 2)) + '</div>';
        return;
      }
      let html = '<h2 class="section-title">Answer</h2><div class="answer">' + esc(j.answer) + '</div>';
      html += '<h2 class="section-title">Sources</h2><div class="sources">';
      (j.sources || []).forEach((s, i) => {
        const pg = (s.page_number != null && s.page_number !== undefined)
          ? '<span class="src-page">Page ' + esc(s.page_number) + '</span>' : '';
        const head = s.section_heading
          ? '<div class="src-head">' + esc(s.section_heading) + '</div>' : '';
        html += '<article class="src"><div class="src-top"><span class="src-name">' + esc(s.source_file) + '</span>'
          + pg + '<span class="score-pill">#' + (i+1) + ' · score ' + esc(s.score) + '</span></div>'
          + head + '<div class="src-preview">' + esc(s.preview) + '</div></article>';
      });
      html += '</div>';
      out.innerHTML = html;
    } catch (err) {
      out.innerHTML = '<div class="err">' + esc(err.message || String(err)) + '</div>';
    } finally {
      btn.disabled = false;
      btn.textContent = prev;
    }
  };
  </script>
</body>
</html>"""


@app.post("/query")
def post_query(body: QueryBody) -> dict:
    """Same payload the CLI uses: answer, sources, retrieved chunk list."""
    try:
        top_k = body.top_k or config.TOP_K
        return answer_query(body.query, top_k=top_k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
def health() -> dict:
    """Load balancers / k8s probes hit this; no dependency checks on purpose."""
    return {"status": "ok"}
