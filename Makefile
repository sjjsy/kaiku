# Makefile for kaiku pip package

# Variables
PACKAGE_NAME := kaiku
DIST_DIR := dist
BUILD_DIR := build

# Default target
all: build

# Build the package
build:
	@echo "Building $(PACKAGE_NAME)..."
	python -m build
	@echo "Build complete. Distribution files are in $(DIST_DIR)/"

# Push the package to PyPI
push:
	@echo "Pushing $(PACKAGE_NAME) to PyPI..."
	twine upload $(DIST_DIR)/*
	@echo "Package pushed to PyPI."

# Clean up build and distribution files
clean:
	@echo "Cleaning up build and distribution files..."
	rm -rf $(BUILD_DIR) $(DIST_DIR) *.egg-info
	@echo "Cleanup complete."

# Run the test suite
test:
	@echo "Running unit tests..."
	pytest tests/unit -v

test-e2e:
	@echo "Running E2E tests (requires whisper-cli on PATH)..."
	pytest tests/e2e -v

test-all:
	@echo "Running all tests..."
	pytest tests/ -v

# Help target
help:
	@echo "Available targets:"
	@echo "  build     - Build the pip package"
	@echo "  push      - Push the package to PyPI"
	@echo "  clean     - Clean up build and distribution files"
	@echo "  test      - Run unit tests"
	@echo "  test-e2e  - Run E2E tests (requires whisper-cli)"
	@echo "  test-all  - Run all tests"
	@echo "  help      - Show this help message"

.PHONY: all build push clean test test-e2e test-all help
