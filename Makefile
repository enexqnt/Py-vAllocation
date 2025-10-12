.PHONY: help test docs examples all

help:
	@echo "Targets:"
	@echo "  test      - run unit tests"
	@echo "  docs      - build Sphinx HTML docs"
	@echo "  examples  - run quick example scripts"
	@echo "  all       - test + docs + examples"

test:
	pytest -q

docs:
	sphinx-build -b html docs docs/_build/html

examples:
	python examples/mean_variance_frontier.py
	python examples/cvar_allocation.py
	python examples/portfolio_ensembles.py

all: test docs examples

