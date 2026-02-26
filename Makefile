.PHONY: help test test-verbose test-cov check


help:
	@echo "Available targets:"
	@echo "  make test         - Run test suite"
	@echo "  make test-verbose - Run test suite with verbose output"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make check        - Run default verification checks"

test:
	PYTHONPATH=src python -m pytest -q

test-verbose:
	PYTHONPATH=src python -m pytest -v

test-cov:
	PYTHONPATH=src python -m pytest --cov=src/rag --cov-report=term-missing

check: test
