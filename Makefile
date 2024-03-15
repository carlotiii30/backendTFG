.DEFAULT_GOAL := help
.PHONY: help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

run: ## Run server
	python -m ./src/main.py

run_test: ## Run tests
	python -m unittest discover -s test

requirements: ## Generate requirements file
	pip freeze > requirements.txt

install: ## Install dependencies
	pip install -r requirements.txt

update: ## Update dependencies
	pip install --upgrade -r requirements.txt