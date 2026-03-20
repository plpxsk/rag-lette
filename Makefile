.PHONY: help test test-verbose test-cov check weaviate


help:
	@echo "Available targets:"
	@echo "  make test         - Run test suite"
	@echo "  make test-verbose - Run test suite with verbose output"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make check        - Run default verification checks"
	@echo "  make weaviate     - Start a local Weaviate container for dev"

test:
	PYTHONPATH=src .venv/bin/python -m pytest -q

test-verbose:
	PYTHONPATH=src .venv/bin/python -m pytest -v

test-cov:
	PYTHONPATH=src .venv/bin/python -m pytest --cov=src/rag --cov-report=term-missing

check: test

weaviate:
	@if docker ps -a --format '{{.Names}}' | grep -qx rag-weaviate; then \
		echo "Starting existing rag-weaviate container..."; \
		docker start rag-weaviate; \
	else \
		echo "Creating rag-weaviate container..."; \
		docker run -d --name rag-weaviate \
			-p 8080:8080 -p 50051:50051 \
			-e QUERY_DEFAULTS_LIMIT=25 \
			-e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
			-e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
			-e DEFAULT_VECTORIZER_MODULE=none \
			cr.weaviate.io/semitechnologies/weaviate:1.30.5; \
	fi
