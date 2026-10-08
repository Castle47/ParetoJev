.PHONY: install test smoke smoke-targets clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest

smoke:
	paretojev synthetic-smoke --config configs/synthetic_smoke.yaml --output artifacts/smoke_metrics.json --checkpoint artifacts/smoke_model.pt

smoke-targets:
	paretojev check-targets artifacts/smoke_metrics.json configs/targets_smoke.yaml --output artifacts/smoke_target_report.json

clean:
	rm -rf .pytest_cache .coverage htmlcov build dist
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
