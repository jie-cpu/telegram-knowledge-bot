# =============================================================================
# Community Platform — Multi-Agent System
# =============================================================================
# Common commands for development, testing, evaluation, and deployment.
# =============================================================================

.PHONY: install lint test test-meta eval run-goal run-scenario watch dry-run
.PHONY: resume list-sessions monitor-heartbeat
.PHONY: db-init db-reset dashboard docker-up docker-down clean security-scan

# ── Installation ──────────────────────────────────────────────────────────

install:
	pip install --upgrade pip
	pip install setuptools anyio
	pip install pydantic pyyaml httpx structlog openai anthropic sqlalchemy alembic asyncpg redis networkx pydantic-settings
	pip install pytest pytest-cov hypothesis syrupy ruff mypy bandit safety pandas streamlit plotly ipython
	@echo "All dependencies installed."
	@echo "Try: make dry-run GOAL=\"Add user profiles\""

install-prod:
	pip install -e "."

# ── Quality ───────────────────────────────────────────────────────────────

lint:
	ruff check agents/ platform/ shared/ tests/ eval/
	ruff format --check agents/ platform/ shared/ tests/ eval/
	mypy agents/ shared/ tests/

format:
	ruff format agents/ platform/ shared/ tests/ eval/

security-scan:
	bandit -r agents/ platform/ shared/ -f json -o reports/security.json
	safety check --full-report

typecheck:
	mypy agents/ shared/

# ── Testing ───────────────────────────────────────────────────────────────

# Run all tests (platform tests + meta-tests of the agent system)
test:
	pytest tests/ -v --tb=short --cov=agents --cov=shared --cov-report=term --cov-report=html

# Run only meta-tests (agent system validation)
test-meta:
	pytest tests/ -v --tb=short

# Run only platform tests (the community web app itself)
test-platform:
	pytest platform/backend/tests/ -v --tb=short

# Run tests with a specific marker
test-mark:
	pytest tests/ -v -m "$(MARK)"

# ── Evaluation ────────────────────────────────────────────────────────────

# Run the full evaluation suite against all golden scenarios
eval:
	python -m eval.runner --all --output-dir reports/eval/

# Run a specific golden scenario
eval-scenario:
	python -m eval.runner --scenario "$(SCENARIO)" --output-dir reports/eval/

# Run with a custom config override (e.g., make eval-config CONFIG=config.prod.yaml)
eval-config:
	python -m eval.runner --all --config "$(CONFIG)" --output-dir reports/eval/

# Generate evaluation report as HTML
eval-report:
	python -m eval.report --input-dir reports/eval/ --output reports/eval_report.html

# ── Running the System ────────────────────────────────────────────────────

# Submit a goal to the orchestrator
run-goal:
	python -m agents.orchestrator --goal "$(GOAL)" --config config.yaml

# Run WITHOUT LLM (rule-based decomposition only, no API key needed)
dry-run:
	python -m agents.orchestrator --goal "$(GOAL)" --dry-run

# Run a pre-built scenario (from eval/datasets/)
run-scenario:
	python -m agents.orchestrator --scenario "eval/datasets/$(SCENARIO).yaml" --config config.yaml

# Watch the agent collaboration in real-time
watch:
	python -m agents.orchestrator --goal "$(GOAL)" --config config.yaml --watch

# Resume a checkpointed session
resume:
	python -m agents.orchestrator --resume "$(SESSION_ID)" --config config.yaml

# List all available sessions
list-sessions:
	python -m agents.orchestrator --list-sessions

# Demo heartbeat monitoring
monitor-heartbeat:
	python -m agents.orchestrator --heartbeat-demo

# ── Database ──────────────────────────────────────────────────────────────

db-init:
	python -m shared.state.init

db-reset:
	rm -f data/state.db
	python -m shared.state.init

db-migrate:
	alembic upgrade head

# ── Dashboard ─────────────────────────────────────────────────────────────

dashboard:
	streamlit run dashboard/app.py --server.port 8501

# ── Docker ────────────────────────────────────────────────────────────────

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-clean:
	docker compose down -v
	docker system prune -f

# ── Monitoring ────────────────────────────────────────────────────────────

# Start local monitoring stack (Prometheus + Grafana)
monitor-up:
	docker compose -f docker-compose.monitoring.yml up -d

monitor-down:
	docker compose -f docker-compose.monitoring.yml down

# ── Utilities ─────────────────────────────────────────────────────────────

clean:
	rm -rf .venv/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf reports/
	rm -rf htmlcov/
	rm -rf *.egg-info/
	rm -rf platform/frontend/node_modules/
	rm -rf platform/frontend/dist/
	rm -rf data/*.db
	rm -rf logs/

# Show token usage for a session
token-usage:
	python -m shared.usage --session "$(SESSION_ID)"

# Show agent costs for a session
cost-report:
	python -m shared.usage --session "$(SESSION_ID)" --costs

# Export session trace for debugging
export-trace:
	python -m shared.state.export --session "$(SESSION_ID)" --output reports/traces/$(SESSION_ID).json

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Installation:"
	@echo "  install           Install all dependencies (dev)"
	@echo ""
	@echo "Testing:"
	@echo "  test              Run all tests with coverage"
	@echo "  test-meta         Run meta-tests of the agent system"
	@echo ""
	@echo "Evaluation:"
	@echo "  eval              Run full evaluation suite"
	@echo "  eval-scenario     Run a specific scenario"
	@echo ""
	@echo "Running:"
	@echo "  run-goal          Submit a goal to the orchestrator"
	@echo "  run-scenario      Run a pre-built scenario"
	@echo "  watch             Watch agent collaboration in real-time"
	@echo "  resume            Resume a checkpointed session"
	@echo ""
	@echo "Dashboard:"
	@echo "  dashboard         Start the Streamlit dashboard"
	@echo "  monitor-up        Start Prometheus + Grafana stack"
	@echo ""
	@echo "Docker:"
	@echo "  docker-up         Start full stack with Docker Compose"
	@echo "  docker-down        Stop Docker stack"
	@echo ""
	@echo "Configuration:"
	@echo "  eval-config       Run eval with custom config file"
