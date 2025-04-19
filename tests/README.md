# Swing Trading Algorithm Tests

This directory contains tests for the Swing Trading Algorithm.

## Test Structure

The tests are organized by module, mirroring the structure of the `src` directory:

```
tests/
├── data/               # Tests for data retrieval and storage
├── analysis/           # Tests for technical analysis and indicators
├── strategy/           # Tests for trading strategies
├── risk/               # Tests for risk management
├── simulation/         # Tests for trade simulation
├── performance/        # Tests for performance tracking
├── optimization/       # Tests for strategy optimization
└── utils/              # Tests for utility functions
```

## Running Tests

To run all tests:

```
python run_tests.py
```

To run tests for a specific module:

```
pytest tests/data
```

To run a specific test file:

```
pytest tests/data/test_models.py
```

To run a specific test:

```
pytest tests/data/test_models.py::TestStock::test_add_data_point
```

## Test Coverage

The tests use pytest-cov to measure code coverage. To see coverage information:

```
pytest --cov=src --cov-report=term-missing
```

## Writing Tests

When writing tests, follow these guidelines:

1. Use descriptive test names that explain what is being tested
2. Use fixtures for common test data
3. Test both normal and edge cases
4. Test error conditions
5. Keep tests independent of each other
6. Use assertions to verify results
