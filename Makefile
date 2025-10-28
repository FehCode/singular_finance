# Singular Finance - Makefile

.PHONY: help install install-dev test lint format clean build docs

help: ## Mostra esta mensagem de ajuda
	@echo "Singular Finance - Comandos Disponíveis:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Instala a biblioteca
	pip install -e .

install-dev: ## Instala dependências de desenvolvimento
	pip install -e ".[dev]"
	pip install -e ".[docs]"

test: ## Executa todos os testes
	pytest tests/ -v --cov=singular_finance --cov-report=html --cov-report=term-missing

test-fast: ## Executa testes rápidos (sem cobertura)
	pytest tests/ -v -m "not slow"

lint: ## Executa linting com flake8
	flake8 singular_finance tests examples

type-check: ## Executa verificação de tipos com mypy
	mypy singular_finance

format: ## Formata código com black
	black singular_finance tests examples

format-check: ## Verifica formatação sem alterar arquivos
	black --check singular_finance tests examples

clean: ## Limpa arquivos temporários
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: ## Constrói o pacote
	python -m build

docs: ## Gera documentação
	cd docs && make html

docs-serve: ## Serve documentação localmente
	cd docs/_build/html && python -m http.server 8000

example: ## Executa exemplo completo
	python examples/exemplo_completo.py

docker-build: ## Constrói imagem Docker
	docker build -t singular-finance .

docker-run: ## Executa container Docker
	docker run -it --rm -v $(PWD):/app singular-finance

docker-test: ## Executa testes no Docker
	docker-compose run tests

jupyter: ## Inicia Jupyter Lab
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

pre-commit: ## Instala pre-commit hooks
	pre-commit install

pre-commit-run: ## Executa pre-commit em todos os arquivos
	pre-commit run --all-files

security: ## Verifica vulnerabilidades de segurança
	pip install safety
	safety check

update-deps: ## Atualiza dependências
	pip install --upgrade pip
	pip install --upgrade -r requirements.txt

check-all: lint type-check format-check test ## Executa todas as verificações

release: clean build ## Prepara release
	twine check dist/*

install-hooks: pre-commit ## Instala todos os hooks de desenvolvimento
