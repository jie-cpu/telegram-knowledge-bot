# Community Platform with Multi-Agent System

A multi-agent orchestrator that deploys **four specialized AI agents** — Software Engineer, Tester, On-Call Monitor, Product Manager — to autonomously build and operate a web platform. The orchestration layer is pure Python (state machine + topological sort + file-based checkpointing); each agent's core intelligence (goal decomposition, code generation, LLM-as-judge) is hand-implemented with LLM + rule-based fallback.

> **Complexity:** Advanced — multi-agent orchestration, role specialization, autonomous task decomposition, LLM-as-judge code evaluation, and long-running (16+ h) session management via file-based checkpointing.

---

## Why This Project Exists

Most AI demos show a single agent in a loop. Real engineering teams have **multiple roles** — developers, QA, on-call, PMs — coordinating on long-lived tasks.

This project demonstrates:

1. **Autonomous task decomposition** — LLM (with rule-based fallback) turns a vague goal into a dependency-ordered task DAG
2. **Testing AI-generated code** — LLM-as-judge scores code on 5 dimensions (correctness, security, quality, error handling, testability) with static-analysis fallback
3. **Long-running agent management** — checkpoint/resume across process restarts, heartbeat monitoring, graceful shutdown

The orchestration layer is pure Python — a hand-written state machine, topological task scheduler, file-based checkpoint manager, and heartbeat monitor. The four agents' core logic (decomposer, code generator, code judge) are independently implemented with LLM + rule-based fallback.

No LangGraph, no CrewAI, no AutoGen. Everything is deliberately hand-written to show understanding at every layer.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                     User / Stakeholder                        │
│              submits a goal (e.g. "Add user profiles")        │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                  Orchestrator (pure Python)                    │
│                                                              │
│   decompose_goal → topological sort → agent dispatch loop     │
│                                         │                    │
│                              ┌──────────┴──────────┐         │
│                              ▼                     ▼         │
│                        execute_task()        check_deps()    │
│                              │                     │         │
│                              └─────→ mark complete ─┘         │
│                                                              │
│   • State: task_states dict (pending → completed)             │
│   • Checkpoint: file-based JSON (after each task)             │
│   • Scheduling: Kahn's algorithm (topological sort)           │
│   • Route: switch on task.agent field                         │
│   • Heartbeat: per-agent health tracking                     │
└───┬──────────┬──────────┬──────────────────┬─────────────────┘
    │          │          │                  │
    ▼          ▼          ▼                  ▼
┌────────┐ ┌────────┐ ┌────────────────┐ ┌────────────────┐
│  SE    │ │ Tester │ │  On-Call       │ │  Product       │
│ Agent  │ │ Agent  │ │  Agent         │ │  Agent         │
│        │ │        │ │                │ │                │
│ LLM    │ │LLM-judge│ │  Simulated    │ │  Rule-based    │
│ coder  │ │static  │ │  deploy +     │ │  PRD writer    │
│        │ │fallback│ │  monitoring   │ │                │
└────────┘ └────────┘ └────────────────┘ └────────────────┘
          │            │
          ▼            ▼
   ┌─────────────────────────┐
   │     Shared State        │
   │  (sqlite state db)      │
   │  + File Checkpoints     │
   │  + Code Artifacts       │
   └─────────────────────────┘
```


## State Machine

Each task follows a 6-state lifecycle with 7 valid transitions:

```
PENDING → ASSIGNED → IN_PROGRESS → REVIEW → ACCEPTED
                                        ↓
                                    REJECTED → IN_PROGRESS (rework)
```

The state machine (`state_machine.py`) enforces these transitions and raises on invalid moves. Rejection count is tracked; after N rejections (configurable), the task escalates to human intervention.

### Topological Scheduling

Tasks form a DAG. The scheduler uses Kahn's algorithm to produce a valid execution order — dependencies always complete before dependents. Cyclic dependencies are detected and rejected at the graph level.

### Checkpoint Architecture

Two-layer persistence:
- **Per-task file checkpoints**: after each completed task, full session state is serialized to `data/checkpoints/` as JSON
- **Rolling window**: configurable max checkpoints per session (default 50)
- **Graceful shutdown**: SIGINT handler saves checkpoint before exit
- **Resume**: `--resume <session_id>` deserializes latest checkpoint, rebuilds state, skips completed tasks

---

## Implementation Status

| Capability | Status | How |
|---|---|---|
| **Autonomous task decomposition** | REAL | `GoalDecomposer` — calls LLM with structured output; falls back to rule-based (keyword detection) when no API key. Returns validated DAG (no cycles, all deps exist). |
| **Testing AI-generated code** | REAL | `LLMJudge` — 5-dimension scoring (correctness, security, quality, error handling, testability). LLM mode scores 1-10 with specific issue locations. Static-analysis fallback detects hardcoded secrets, missing validation, absent error handling. |
| **Long-running agent management** | REAL | `SessionManager` + `CheckpointManager` — JSON checkpoint after each task, rolling window, `--resume`, SIGINT handler, heartbeat monitoring demo. |
| **SE Agent writes code to disk** | REAL | `SECoder` — LLM generates FastAPI code files, writes to `platform/backend/app/`. Context reader adapts to existing project structure. Falls back to simulated output without API key. |
| **On-Call Agent** | Simulated | Prints deploy + health check + monitor steps. |
| **PM Agent** | Rule-based | Generates PRD from task acceptance criteria. |
| **Dashboard** | Removed | Was a stub with hardcoded data, no value for interview. |
| **Frontend** | Scaffold | `package.json` exists, no components implemented. |

---

## Project Structure

```
agents/
├── orchestrator/               # State machine + topological sort + checkpoint
│   ├── __main__.py             # CLI entry point
│   ├── decomposer.py           # Goal → task DAG (LLM + rule fallback)  [REAL]
│   ├── scheduler.py            # Topological sort (Kahn's algorithm)
│   └── state_machine.py        # Task lifecycle states
├── se_agent/
│   └── coder.py                # LLM code generation → file write  [REAL]
├── tester_agent/
│   └── llm_judge.py            # 5-dimension code evaluation  [REAL]
├── pm_agent/                   # Rule-based PRD generation
└── oncall_agent/               # Simulated deploy/monitor

shared/
├── llm_client.py               # Multi-provider LLM client (OpenAI, Anthropic, DeepSeek, Gemini, Ollama)
└── state/
    ├── session_manager.py      # Checkpoint/resume, heartbeat, graceful shutdown  [REAL]
    ├── checkpoint.py           # File-based checkpoint with rolling window  [REAL]
    ├── models.py               # SessionState, TaskGraph data classes
    └── init.py                 # SQLite DB initialization

tests/
├── conftest.py                 # Shared fixtures
├── test_orchestrator_decomposition.py
└── test_long_running.py

eval/
└── datasets/
    └── add-user-profiles.yaml

config.yaml                     # Single config point for all agents + providers
```

---

## Quick Start

### Prerequisites

- Python 3.12+
- OpenAI / Anthropic / DeepSeek / Gemini API key (optional — dry-run mode works without one)

### Setup

```bash
# 1. Install dependencies
make install

# 2. Run a demo (no API key needed — rule-based + simulated)
make dry-run GOAL="Add a user profile page with avatar upload"

# 3. Run with LLM (set API key in .env)
echo 'OPENAI_API_KEY=sk-...' > .env
make run-goal GOAL="Add a user profile page with avatar upload"
```

### Commands

```bash
make dry-run GOAL="Add user profiles"          # Rule-based, no API key
make run-goal GOAL="Add user profiles"         # LLM mode (requires API key)
make list-sessions                             # List saved sessions
make resume SESSION_ID=sess_...                # Resume from checkpoint
make monitor-heartbeat                         # Heartbeat monitoring demo
make db-init                                   # Initialize state DB
make eval                                      # Run evaluation suite
make test-meta                                 # Run meta-tests
```

---

## Configuration

[`config.yaml`](config.yaml) is the single control point. Change any parameter and re-run:

```yaml
llm:
  default_provider: "openai"       # or "anthropic" | "deepseek" | "gemini" | "ollama"
  providers:
    openai:
      api_key: "${OPENAI_API_KEY}"
      default_model: "gpt-4o"
    deepseek:
      api_key: "${DEEPSEEK_API_KEY}"
      default_model: "deepseek-chat"
    gemini:
      api_key: "${GEMINI_API_KEY}"
      default_model: "gemini-2.5-flash"
```

Full documentation is inline in `config.yaml`. Every agent has tunable parameters (model, temperature, thresholds).

---

## Testing & Evaluation

### Meta-Tests

| Test | What it verifies |
|---|---|
| `test_orchestrator_decomposition.py` | Goal decomposition → valid DAG, state machine transitions (6 states, 7 transitions), rejection escalation, agent handoff |
| `test_long_running.py` | Checkpoint serialization roundtrip, process restart recovery, rolling window retention, latest-checkpoint lookup |

### LLM-as-Judge Scoring Dimensions

| Dimension | Weight | What it catches |
|---|---|---|
| Correctness | 30% | Does the code implement what was asked? |
| Security | 25% | Hardcoded secrets, missing input validation, SQL injection |
| Code Quality | 20% | Type hints, docstrings, naming, structure |
| Error Handling | 15% | try/except, HTTPException, user-facing messages |
| Testability | 10% | Dependency injection, clean interfaces |

Two modes: **LLM** (scores 1-10 with location-specific issues, requires API key) or **static analysis** (keyword-based detection, no API key needed).

---

## Long-Running Agent Management

### File Checkpoints

- Saved after every completed task to `data/checkpoints/`
- Rolling window: keeps N most recent (default 50) per session
- JSON format: goal, task graph, iteration count, estimated cost
- `--resume <session_id>` deserializes and continues from last checkpoint
- SIGINT handler saves checkpoint before exit

### Heartbeat Monitoring

```bash
make monitor-heartbeat
```

Tracks 4 agents + orchestrator. States: `alive` → `missed_beats` → `dead`. Configurable thresholds for missed beat count and interval.

---

## Hiring Manager FAQ

**Q: What makes this different from AutoGPT or simple agent loops?**

This models an *organization*, not a single agent. Four specialized agents with distinct roles, tools, and handoff protocols. The orchestrator is a hand-written state machine (6 states, 7 valid transitions) + topological sort (Kahn's algorithm) + file-based checkpointing.

**Q: Why pure Python instead of LangGraph / CrewAI / AutoGen?**

Those frameworks abstract away the state management, checkpointing, and routing logic. In this project, those are the interesting parts — the state machine, dependency graph, checkpoint serialization, and heartbeat monitor are all deliberately hand-written to show understanding. Each module can be swapped for a framework later without changing the agent implementations (decomposer, coder, judge).

**Q: How do you prevent infinite loops?**

Three mechanisms: (1) iteration limit per task (configurable), (2) escalation threshold — 5 revision loops between SE and Tester triggers pause, (3) task-level timeout that resets context on expiry.

**Q: How do you evaluate AI-generated code?**

Two-tier gate: (a) LLM-as-judge scores code on 5 dimensions (correctness, security, quality, error handling, testability) — gives 1-10 per dimension with specific issue locations; (b) static analysis fallback detects hardcoded secrets, missing validation, absent error handling, missing type hints. Both produce a pass/fail decision against a configurable threshold.

**Q: How do you handle 16+ hour runs?**

File-based checkpointing via `CheckpointManager`. After each completed task, full session state is serialized to `data/checkpoints/` as a JSON file with rolling window retention (default 50). `SessionManager` handles lifecycle — `--resume` reloads the latest checkpoint, rebuilds state, skips completed tasks. SIGINT triggers immediate checkpoint save before exit.

**Q: Can you run this locally without cloud APIs?**

Yes. `make dry-run` uses rule-based decomposition and static-analysis judging — no API calls. All tasks in the goal lifecycle complete without internet access.

**Q: What actually writes real code?**

The SE Agent (LLM mode, requires API key) generates FastAPI code files and writes them to `platform/backend/app/`. The code includes async endpoints, Pydantic models, proper error handling, type hints, and docstrings. Without an API key, it falls back to simulated output.

---

## License

[MIT](LICENSE)
