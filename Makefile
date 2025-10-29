.PHONY: help build clean publish test

help:
	@echo "Available commands:"
	@echo "  make build     - Build the package"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make publish   - Publish package to PyPI"
	@echo "  make test      - Run tests"

build:
	uv build

clean:
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

publish: build
	@if [ ! -f .env ]; then \
		echo "Error: .env file not found"; \
		exit 1; \
	fi
	@source .env && uv publish

test:
	uv run pytest
