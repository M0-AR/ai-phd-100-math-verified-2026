test:
	python -m pytest -q
run:
	python experiments/run_all.py
live:
	python experiments/live_market_verify.py
patterns:
	python experiments/hidden_patterns.py
all: test run live patterns
screenshots:
	python scripts/capture.py
