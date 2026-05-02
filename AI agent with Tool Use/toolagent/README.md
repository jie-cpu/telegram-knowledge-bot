# ToolAgent — An Extensible AI Agent with Tool Use

[![CI](https://github.com/YOUR_USERNAME/toolagent/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/toolagent/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI agent system that uses multiple tools to accomplish tasks. The agent follows a **ReAct (Reasoning + Acting)** loop: it reasons about the current state, decides which tool to call, executes the tool, observes the result, and iterates until the task is complete.

> **Portfolio Project** — This project demonstrates the skills required for AI Engineer roles involving agent-based systems. Agents appear in **14.4% of AI Engineer job listings** and the demand is growing.

---

## ✨ Features

### 🔧 Tool System
- **7 built-in tools**: Calculator, Web Search, Database Query, Code Execution, Weather, File Manager, Send Email
- Each tool has **JSON Schema** input/output definitions
- Unified `BaseTool` abstract class for easy extension
- OpenAI and Anthropic tool format support

### 🔄 Agent Loop (ReAct Pattern)
- **Step-limited reasoning** with configurable maximum steps
- **Timeout protection** at per-step and per-session levels
- Automatic error recovery and retry
- Streaming callback support for real-time step observation
- **No external frameworks** — hand-written ReAct implementation for full control and understanding

### 🛡️ Guardrails
- **Input guardrails**: Length validation, content safety, PII detection, prompt injection prevention
- **Output guardrails**: Tool call validation, business rule enforcement, rate limiting
- **Security**: Dangerous code detection, SQL injection prevention, path traversal protection

### 📊 Observability
- **Trace logging**: Hierarchical span-based tracing with JSON output
- **Cost tracking**: Per-session and per-token cost instrumentation
- **Metrics collection**: Tool latency, success rates, step counts
- **LLM cost estimation** before making API calls

### 🔬 Evaluation
- **Golden dataset** with 11+ test scenarios
- **Multi-step evaluation** — measures correct tool selection AND correct sequence
- **F1-based scoring** for tool selection accuracy
- **Tool-wise accuracy breakdown**

### 🚀 Deployment
- **FastAPI** REST API with interactive Swagger docs
- **CLI** for interactive sessions and automation
- **Docker** ready
- **GitHub Actions** CI/CD pipeline

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Input                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                     Guardrails (Input)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Length Check │  │ Content Safe │  │  PII Detect  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Agent Loop (ReAct)                        │
│                                                              │
│  ┌─────────┐    ┌──────────┐    ┌───────────┐              │
│  │  LLM    │───▶│  Select  │───▶│  Execute  │              │
│  │ Reason  │    │  Tool    │    │   Tool    │              │
│  └─────────┘    └──────────┘    └───────────┘              │
│       ▲                              │                      │
│       └──────────────────────────────┘                      │
│                    (iterate)                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                     Guardrails (Output)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Tool Validate│  │  Rate Limit  │  │Business Rules│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   Final Response                             │
└─────────────────────────────────────────────────────────────┘

         ┌──────────────────────────────────────┐
         │         Observability Layer          │
         │  Trace Logs  │  Metrics  │  Cost     │
         └──────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- (Optional) OpenAI or Anthropic API key for real LLM calls

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/toolagent.git
cd toolagent

# Install with dev dependencies
pip install -e ".[dev]"

# Or using make
make install
```

### Run in Interactive Mode

```bash
# Start interactive session (uses mock LLM — no API key needed)
toolagent
```

This starts an interactive chat session. The mock LLM simulates tool selection based on keyword matching, so the system works **out of the box** without any API keys.

### Run a Single Query

```bash
toolagent --prompt "What's the weather in Tokyo?"
toolagent --prompt "Calculate 15 * 3 / 5" --json
```

### List Available Tools

```bash
toolagent --list-tools
```

### Start the API Server

```bash
toolagent --api --port 8000
```

Then visit `http://localhost:8000/docs` for interactive Swagger documentation.

---

## 🛠️ Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `calculator` | Evaluate mathematical expressions | `expression` (required): string |
| `web_search` | Search the web for information | `query` (required): string, `num_results` (optional): integer |
| `database_query` | Query the company database (products, customers, orders) | `query` (required): string, `format` (optional): "json" or "csv" |
| `code_executor` | Execute Python code in a sandbox | `code` (required): string, `timeout` (optional): integer |
| `weather` | Get current weather and 3-day forecast for a city | `city` (required): string, `units` (optional): "metric" or "imperial" |
| `file_manager` | Read, write, list, and delete files in a sandboxed directory | `operation` (required): "read", "write", "list", "delete", "info"; `path` (required): string; `content` (optional): string |
| `send_email` | Send an email to recipients | `to` (required): string[], `subject` (required): string, `body` (required): string, `cc` (optional): string[] |

### Adding a New Tool

```python
from agent.tools.base import BaseTool

class MyCustomTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "Describe what this tool does and when to use it"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "Description of param1",
                },
            },
            "required": ["param1"],
        }

    async def _run(self, param1: str) -> dict:
        # Your tool logic here
        return {"result": f"Processed: {param1}"}

# Register with the agent
registry.register(MyCustomTool())
```

---

## 🛡️ Guardrails

Guardrails protect the agent from misuse and enforce business rules:

### Input Guardrails

| Guardrail | What it blocks |
|-----------|---------------|
| `InputLengthGuardrail` | Inputs exceeding max length (default: 10,000 chars) |
| `InputContentGuardrail` | Prompt injection attempts, dangerous system commands |
| `PIIGuardrail` | Credit card numbers, Social Security Numbers |

### Output Guardrails

| Guardrail | What it validates |
|-----------|------------------|
| `ToolCallValidationGuardrail` | Safe code (blocks dangerous imports), valid SQL (blocks DROP/TRUNCATE), valid emails, no path traversal |
| `RateLimitGuardrail` | Maximum requests per minute |
| `BusinessRuleGuardrail` | Custom business rules (max values, allowed values) |

---

## 📊 Observability

### Trace Logging

Structured trace logs are written to `logs/traces.jsonl`:

```json
{"type": "tool_call", "tool": "calculator", "args": {"expression": "2+2"}, "success": true, "duration_ms": 1.23}
{"type": "llm_call", "provider": "mock", "model": "mock-model", "input_tokens": 45, "output_tokens": 12, "cost_usd": 0.0, "latency_ms": 5.0}
{"type": "guardrail_check", "guardrail": "InputContentGuardrail", "passed": true}
```

### Metrics

View system metrics via the API:

```bash
curl http://localhost:8000/metrics
```

### Cost Tracking

Track LLM API costs across sessions:

```bash
curl http://localhost:8000/costs
```

```json
{
  "total_cost_usd": 0.001234,
  "total_sessions": 5,
  "daily_costs": {"2025-04-15": 0.001234}
}
```

---

## 🔬 Evaluation

The project includes a comprehensive evaluation harness for measuring agent performance.

### Running Evaluation

```bash
python -m demo.evaluation_demo
```

### Golden Dataset

The golden dataset (`tests/evaluation/dataset.py`) includes tests for:

- **Single tool calls**: Calculator, weather, search, database, email, code
- **Multi-tool sequences**: Weather then search, database then calculate
- **Guardrail effectiveness**: Prompt injection blocking
- **Tool sequence ordering**: Correct tool ordering verification

### Scoring Methodology

- **Tool Selection Score**: F1 score between expected and actual tools used
- **Sequence Score**: Proportion of tools appearing in the correct order
- **Pass/Fail**: No missed tools + selection score >= 0.5 + no errors

### Example Evaluation Output

```
════════════════════════════════════════════════════════
EVALUATION REPORT: ToolAgent Golden Dataset
════════════════════════════════════════════════════════

📊 Overall Results:
  Total Examples:  11
  Passed:          10
  Failed:          1
  Pass Rate:       90.9%
  Avg Selection:   95.5%
  Avg Sequence:    97.0%
  Avg Steps:       2.5
  Total Cost:      $0.000000
  Total Tokens:    492

🔧 Tool Accuracy:
  calculator: 100.0% (3/3)
  weather: 100.0% (2/2)
  web_search: 100.0% (2/2)
  database_query: 100.0% (3/3)
  send_email: 100.0% (1/1)
  code_executor: 100.0% (1/1)
```

---

## 🌐 API Reference

Start the API server:

```bash
toolagent --api
```

### Chat Endpoint

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in Tokyo?", "max_steps": 10}'
```

### List Tools

```bash
curl http://localhost:8000/tools
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Metrics

```bash
curl http://localhost:8000/metrics
```

---

## 📁 Project Structure

```
toolagent/
├── agent/
│   ├── core/              # Agent loop, LLM client, memory
│   │   ├── agent.py       # Main ReAct agent loop
│   │   ├── llm_client.py  # OpenAI/Anthropic/Mock providers
│   │   └── memory.py      # Conversation memory management
│   ├── tools/             # Tool definitions (7 tools)
│   │   ├── base.py        # Abstract base tool class
│   │   ├── registry.py    # Tool registration and dispatch
│   │   ├── calculator.py
│   │   ├── web_search.py
│   │   ├── database_query.py
│   │   ├── code_executor.py
│   │   ├── weather.py
│   │   ├── file_manager.py
│   │   └── send_email.py
│   ├── guardrails/        # Input/output validation
│   │   ├── base.py
│   │   ├── input_guardrails.py
│   │   └── output_guardrails.py
│   └── observability/     # Tracing, metrics, cost
│       ├── tracer.py
│       ├── metrics.py
│       └── cost_tracker.py
├── api/                   # FastAPI REST API
│   ├── main.py            # API routes and app setup
│   └── models.py          # Request/response schemas
├── cli/                   # Command-line interface
│   └── main.py            # Interactive and single-query modes
├── tests/                 # Comprehensive test suite
│   ├── test_tools.py      # Unit tests for all tools
│   ├── test_agent.py      # Agent loop tests
│   ├── test_guardrails.py # Guardrail tests
│   └── evaluation/        # Evaluation harness
│       ├── dataset.py     # Golden evaluation dataset
│       ├── evaluator.py   # Scoring and reporting
│       └── scenarios.py   # Multi-step scenarios
├── demo/                  # Demo scripts
│   ├── basic_usage.py     # Feature demonstrations
│   └── evaluation_demo.py # Evaluation run
├── config/
│   └── settings.py        # Configuration management
├── .github/workflows/     # CI/CD pipeline
├── README.md
├── pyproject.toml
├── Makefile
├── .env.example
└── requirements.txt
```

---

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v --cov=agent --cov-report=term-missing
```

### Run Specific Test Categories

```bash
# Tool tests only
pytest tests/test_tools.py -v

# Agent loop tests only
pytest tests/test_agent.py -v

# Guardrail tests only
pytest tests/test_guardrails.py -v

# Quick test (stop on first failure)
make test-quick
```

### Test Coverage

The test suite covers:
- ✅ All 7 tools with multiple test cases each
- ✅ Agent initialization and basic execution
- ✅ Mock LLM integration
- ✅ Input and output guardrails
- ✅ Business rule enforcement
- ✅ Guardrail blocking behavior
- ✅ Session metadata and tracking
- ✅ Evaluation harness scoring

---

## ⚙️ Configuration

Configuration is managed via environment variables (see `.env.example`):

```bash
# Use mock LLM (no API key needed — runs out of the box)
LLM_PROVIDER=mock

# Or use OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-...

# Or use Anthropic
LLM_PROVIDER=anthropic
ANTHROPIC_MODEL=claude-sonnet-4-20250514
ANTHROPIC_API_KEY=sk-ant-...
```

---

## 💡 Skills Demonstrated

| Skill | Implementation |
|-------|---------------|
| **Tool Definition & Schema Design** | Each tool extends `BaseTool` with JSON Schema `parameters`, OpenAI/Anthropic format converters |
| **Agent Loop with Step Limits & Timeouts** | Hand-written ReAct loop in `agent.py` with `max_steps`, `step_timeout_seconds`, `session_timeout_seconds` |
| **Guardrails & Business Rule Constraints** | Input guardrails (length, content, PII), output guardrails (tool validation, rate limiting, business rules) |
| **Multi-Step Evaluation** | Golden dataset with 11 examples, F1-based tool selection scoring, sequence ordering verification |
| **Cost & Latency Instrumentation** | `CostTracker` with per-session/per-token costs, `MetricsCollector` with tool latencies and p95 |
| **Trace Logging & Observability** | `TraceLogger` with hierarchical spans, structured JSON logs, span-based event tracking |

---

## 📈 Interview Tips

This project is designed for AI Engineer interviews. Here's what hiring managers look for:

1. **Clear README** — This file is the first thing they read. It explains the problem, solution, and architecture.
2. **Tests** — The project includes unit tests, integration tests, and evaluation tests. Run them with `pytest`.
3. **Evaluation** — The golden dataset and evaluation harness demonstrate quantitative thinking about agent performance.
4. **Guardrails** — Security and safety features show production awareness.
5. **Observability** — Logging, metrics, and cost tracking demonstrate operations readiness.
6. **Clean Code** — Modular architecture with clear separation of concerns.

### During the Interview

- **Walk through the architecture**: Start with the README diagram, explain the ReAct loop, then show how tools are defined.
- **Discuss the evaluation**: Show the golden dataset and explain how you measure tool selection accuracy.
- **Talk about guardrails**: Explain why you added input/output validation and how it prevents misuse.
- **Mention trade-offs**: Mock vs real LLM, sandbox security, step limits vs task complexity.

---

## 📚 References

- [AI Engineering Field Guide — Portfolio Projects](https://github.com/alexeygrigorev/ai-engineering-field-guide/tree/main/portfolio)
- [Anthropic Tool Use Documentation](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

---

## 📄 License

MIT License
