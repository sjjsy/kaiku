# Makefile for kaiku

PACKAGE_NAME := kaiku
DIST_DIR := dist
BUILD_DIR := build

.PHONY: all build clean test test-e2e test-all help

all: build

# Local build only; PyPI publish is tag-driven via .github/workflows/publish.yml
build:
	@echo "Building $(PACKAGE_NAME)..."
	uv build --clear
	@echo "Build complete. Distribution files are in $(DIST_DIR)/"

clean:
	@echo "Cleaning up build and distribution files..."
	rm -rf $(BUILD_DIR) $(DIST_DIR) *.egg-info
	@echo "Cleanup complete."

test:
	@echo "Running unit tests..."
	pytest tests/unit -v

test-e2e:
	@echo "Running E2E tests (requires whisper-cli on PATH)..."
	pytest tests/e2e -v

test-all:
	@echo "Running all tests..."
	pytest tests/ -v

help:
	@echo "Available targets:"
	@echo "  build     - Build sdist + wheel locally (uv build)"
	@echo "  clean     - Clean up build and distribution files"
	@echo "  test      - Run unit tests"
	@echo "  test-e2e  - Run E2E tests (requires whisper-cli)"
	@echo "  test-all  - Run all tests"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Publish: git tag vX.Y.Z && git push origin vX.Y.Z  (GitHub Actions + PyPI trusted publishing)"
