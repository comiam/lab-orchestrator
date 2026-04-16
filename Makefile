.PHONY: lint format check test

SRC = lab_orchestrator tests examples

lint:
	uv run ruff check $(SRC)
	uv run flake8 $(SRC)
	uv run mypy lab_orchestrator

format:
	uv run ruff format $(SRC)
	uv run ruff check --fix $(SRC)

check: lint test

test:
	uv run pytest tests/ -v
