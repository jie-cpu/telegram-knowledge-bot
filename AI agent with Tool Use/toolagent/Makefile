.PHONY: install test lint run api clean demo eval

# ── Installation ───────────────────────────────────────────────────
install:
	pip install -e ".[dev]"

install-minimal:
	pip install -e .

# ── Development ────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=agent --cov-report=term-missing

test-quick:
	pytest tests/ -v --timeout=30 -x

lint:
	ruff check . --fix

typecheck:
	mypy agent/ --ignore-missing-imports

format:
	ruff format .

# ── Running ────────────────────────────────────────────────────────
run:
	toolagent

api:
	toolagent --api

# ── Demos ──────────────────────────────────────────────────────────
demo:
	python -m demo.basic_usage

eval:
	python -m demo.evaluation_demo

# ── Cleanup ────────────────────────────────────────────────────────
clean:
	rm -rf logs/ traces.jsonl
	rm -rf .pytest_cache/ __pycache__/
	rm -rf htmlcov/ .coverage
	rm -rf *.egg-info/ dist/ build/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
