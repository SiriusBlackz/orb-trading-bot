#!/bin/bash
# Run ORB Trading Bot system tests
# Usage: ./tests/run_tests.sh [options]
#
# Options:
#   --all       Run all tests including TWS connection tests
#   --no-tws    Skip tests requiring TWS connection
#   --quick     Run only import and environment tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=============================================="
echo "ORB Trading Bot - System Tests"
echo "=============================================="
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Check pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "Installing pytest..."
    pip install pytest
fi

# Parse arguments
RUN_MODE="all"
PYTEST_ARGS="-v --tb=short"

for arg in "$@"; do
    case $arg in
        --all)
            RUN_MODE="all"
            ;;
        --no-tws)
            RUN_MODE="no-tws"
            PYTEST_ARGS="$PYTEST_ARGS --ignore-glob='*TWS*' -k 'not tws and not connect and not fetch'"
            ;;
        --quick)
            RUN_MODE="quick"
            PYTEST_ARGS="$PYTEST_ARGS -k 'Import or Environment'"
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--all|--no-tws|--quick]"
            exit 1
            ;;
    esac
done

echo "Run mode: $RUN_MODE"
echo ""

# Run tests based on mode
if [ "$RUN_MODE" = "all" ]; then
    echo "Running ALL tests (requires TWS connection)..."
    echo "----------------------------------------------"
    echo ""
    pytest tests/test_setup.py $PYTEST_ARGS
elif [ "$RUN_MODE" = "no-tws" ]; then
    echo "Running tests WITHOUT TWS connection..."
    echo "----------------------------------------------"
    echo ""
    pytest tests/test_setup.py $PYTEST_ARGS \
        -k "not (TWSConnection or HistoricalData)"
elif [ "$RUN_MODE" = "quick" ]; then
    echo "Running QUICK tests (imports and environment)..."
    echo "----------------------------------------------"
    echo ""
    pytest tests/test_setup.py $PYTEST_ARGS \
        -k "Imports or Environment"
fi

echo ""
echo "=============================================="
echo "Tests completed!"
echo "=============================================="
