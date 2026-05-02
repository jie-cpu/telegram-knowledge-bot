.PHONY: dev install lint test run docker-build docker-run clean

# ---- Development ----

install:
	pip install -e .[dev]

dev:
	python -m src.main

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/

# ---- Testing ----

test:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing

test-ci:
	python -m pytest tests/ -v --cov=src --cov-report=xml --cov-report=term

# ---- Docker ----

docker-build:
	docker compose build

docker-run:
	docker compose up -d

docker-stop:
	docker compose down

# ---- Cleanup ----

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage coverage.xml
