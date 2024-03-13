.DEFAULT_GOAL := help
.PHONY: help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

test: ## Run pytest tests
	pytest -v

requirements: ## Generate requirements file
	pip freeze > requirements.txt

dependencies: ## Install dependencies
	pip install -r requirements.txt