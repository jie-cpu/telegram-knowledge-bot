# Personal Knowledge Management Bot

A Telegram bot that processes **voice notes, images, links, and text** through an **async agent pipeline** and organises everything into structured articles stored in a **git-backed knowledge base**.

**Complexity:** Intermediate — multi-modal input, agent orchestration with skills/commands pattern, parallel subagents, rate limiting, session persistence.

---

## The Problem

We consume information across multiple formats every day — voice notes, screenshots, articles, quick thoughts. None of it gets organised. This bot is a **personal knowledge base with an AI editor**: send it anything, and it returns a structured, searchable, version-controlled article.

---

## Quick Start

```
pip install -e . && pip install faster-whisper
```

Copy `.env.example` to `.env` and fill in:

```
TELEGRAM_BOT_TOKEN=...
DEEPSEEK_API_KEY=...
```

```
python -m src.main
```

Talk to your bot on Telegram. That's it.

> Voice transcription runs **locally** via `faster-whisper` — no API key needed for voice. The LLM (DeepSeek/OpenAI/Anthropic) is only used for article generation.

---

## How It Works

### Data Flow

```
Telegram → Handler → Processor ──→ Orchestrator Pipeline ──→ Git Knowledge Base
                           │       ┌──────────────────┐         │
                    ┌──────┼──────┐│  classify        │         │
                    ▼      ▼      ▼│    │              │         ▼
               Voice   Image   Link │    ▼              │    data/knowledge/
              (faster-whisper)      │  extract          │    └─ 2026/05/
               (Vision API)         │    │              │       └─ <id>_<title>.md
              (trafilatura)         │    ▼              │
               Text (direct)        │  subagents ──→ generate ──→ link
                                   │              ↑
                                   │        fallback (no LLM key)
                                   └──────────────────┘
```

### Pipeline Steps ([src/agent/orchestrator.py](src/agent/orchestrator.py))

| Step | What it does | Parallelism |
|------|-------------|-------------|
| `classify` | Keyword-based categorisation → technology, science, personal, etc. | Sequential |
| `extract` | Regex-based named entity extraction → capitalized phrases | Sequential |
| `subagents` | 3 agents: entity enrichment, research context, content formatting | **Parallel** (`asyncio.gather`) |
| `generate` | Calls DeepSeek/OpenAI/Anthropic with a structured prompt → JSON article | Conditional (if LLM key exists) |
| `fallback` | Pure-skill article assembly using `SummariseSkill` | Alternative branch |
| `link` | Keyword overlap matching against existing KB articles → related links | Sequential |

Each step is an async function — clear parameters in, structured result out. No graph framework.

### Multi-Modal Processors

| Input | Processor | Key Code |
|-------|-----------|----------|
| 🎤 Voice | `faster-whisper` (local) or OpenAI Whisper API | [`src/processing/voice.py`](src/processing/voice.py) |
| 🖼 Image | DeepSeek / OpenAI / Anthropic vision API | [`src/processing/vision.py`](src/processing/vision.py) |
| 🔗 Link | `trafilatura` content extraction | [`src/processing/link.py`](src/processing/link.py) |
| 📝 Text | Metadata analysis (word count, code detection, question detection) | [`src/processing/text.py`](src/processing/text.py) |

### Agent Architecture ([src/agent/orchestrator.py](src/agent/orchestrator.py))

```python
async def process(self, content, user_id) -> Article | None:
    category = await ClassifySkill().execute(ctx, text=text)
    entities = await ExtractSkill().execute(ctx, text=text)
    sub_results = await SubagentPool().run_all(content)       # parallel
    article = await self._generate_with_llm(...)              # DeepSeek
              or await self._fallback_generate(...)
    article = await LinkSkill().execute(ctx, text=article.text)
    return article
```

The orchestrator is **hand-written async** (~80 lines). No LangChain, no LangGraph — just Python `asyncio` and direct `httpx` calls. This makes the data flow explicit and debuggable.

---

## Skills Demonstrated

### AI Engineering

| Skill | Evidence |
|-------|----------|
| **Multi-modal input handling** | Voice transcription, vision analysis, link extraction, text analysis — each in its own module |
| **Agent orchestration** | Skills/commands pattern: `ClassifySkill`, `ExtractSkill`, `LinkSkill`, `SummariseSkill` |
| **Parallel subagents** | `EntityEnrichment`, `Research`, `Formatting` agents run via `asyncio.gather` |
| **LLM provider abstraction** | DeepSeek/OpenAI/Anthropic switchable via one config line |
| **Structured output from LLM** | Prompt-engineered JSON generation with markdown fence parsing fallback |

### Production Engineering

| Skill | Evidence |
|-------|----------|
| **Rate limiting** | Per-user token-bucket (`TokenBucket` in [`src/queue/rate_limiter.py`](src/queue/rate_limiter.py)) |
| **Session persistence** | SQLite with WAL mode — survives restarts ([`src/storage/session.py`](src/storage/session.py)) |
| **Git as a knowledge base** | Every article is a markdown file committed with full version history ([`src/storage/git_store.py`](src/storage/git_store.py)) |
| **Graceful shutdown** | Signal handlers (SIGINT/SIGTERM) for clean teardown |
| **Docker deployment** | `Dockerfile` + `docker-compose.yml` |
| **CI/CD** | GitHub Actions — lint, test, coverage on every push |

---

## Evaluation & Testing

Every project should have eval. This one has **18 unit tests**:

```
tests/
├── test_agent.py       # 10 tests — skills (classify, extract, summarise, link) + subagents
├── test_processing.py  # 5 tests — text analysis, article model, markdown rendering
└── test_queue.py       # 3 tests — rate limiter, priority queue, retry logic
```

```bash
make test        # pytest + coverage
make lint        # ruff
```

Test patterns used: async test fixtures, mock-free unit tests (skills use pure logic), parametrised edge cases (empty text, code blocks, lists).

> The CI pipeline (`.github/workflows/ci.yml`) runs lint + test + coverage across Python 3.11 and 3.12 on every push.

---

## Project Structure

```
src/
├── main.py                 # Entry point, wiring, graceful shutdown
├── config.py               # pydantic-settings (auto-loads .env)
├── agent/
│   ├── orchestrator.py     # Core pipeline: classify → extract → subagents → generate → link
│   ├── skills.py           # ClassifySkill, ExtractSkill, LinkSkill, SummariseSkill
│   └── subagents.py        # EntityEnrichmentAgent, ResearchAgent, FormattingAgent
├── bot/
│   └── handlers.py         # Telegram message/command handlers
├── models/
│   └── schemas.py          # Pydantic models — IncomingMessage, ProcessedContent, Article
├── processing/
│   ├── voice.py, vision.py, link.py, text.py
├── queue/
│   └── rate_limiter.py     # Token bucket per user
└── storage/
    ├── session.py          # SQLite — user sessions, article index, processing log
    └── git_store.py        # Git-backed markdown articles
```

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| **Hand-written async, not LangGraph** | 6 linear steps don't need a graph framework. Function calls are clearer, debug with normal tracebacks, zero extra dependencies |
| **Local faster-whisper** | Zero per-use cost for voice. Works fully offline after model download. Falls back gracefully if model not installed |
| **DeepSeek as default** | OpenAI-compatible API, competitive pricing, vision support. Swapping to OpenAI is one env var change |
| **Direct httpx, not LangChain** | Full control over requests, no framework churn, easier to debug |
| **SQLite + Git** | Zero infrastructure. SQLite for sessions, Git for articles = backup and history for free |
| **3 parallel subagents** | Entity enrichment, research context, and content formatting run concurrently. Each is independent so parallelism is safe |

---

## Production Upgrades (If This Were a Real Product)

| Current | Production |
|---------|-----------|
| In-process rate limiter | Redis-backed sliding window |
| SQLite | PostgreSQL + pgvector |
| Local whisper | Dedicated ASR service (faster + multi-language) |
| Single process | Kubernetes deployment with message queue (Redis/SQS) |
| — | Prometheus metrics + structured logging to Datadog/Grafana |

---

## Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome + feature overview |
| `/help` | Detailed help |
| `/recent` | Last 10 articles |
| `/search <query>` | Full-text search across KB |
| `/stats` | Article count, commits, storage |
| `/export` | Push git repo to remote |

---

## All Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | — | Required |
| `DEEPSEEK_API_KEY` | — | DeepSeek (default provider) |
| `DEEPSEEK_MODEL` | `deepseek-chat` | |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | |
| `OPENAI_API_KEY` | — | Alternative provider |
| `ANTHROPIC_API_KEY` | — | Alternative provider |
| `LLM_PROVIDER` | `auto` | `deepseek`, `openai`, `anthropic` |
| `TRANSCRIPTION_MODE` | `local` | `local` or `api` |
| `WHISPER_MODEL_SIZE` | `base` | `tiny`/`base`/`small`/`medium`/`large` |
| `RATE_LIMIT_MESSAGES` | `10` | |
| `RATE_LIMIT_WINDOW` | `60` | Seconds |

---

## License

MIT
