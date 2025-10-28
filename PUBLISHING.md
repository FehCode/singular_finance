# Publicação da biblioteca Singular Finance

Este documento descreve um fluxo seguro e reprodutível para publicar a biblioteca no TestPyPI e no PyPI.

Pré-requisitos
- Python >= 3.8
- Ter uma conta no TestPyPI (https://test.pypi.org) e no PyPI (https://pypi.org)
- Ter `twine` e `build` instalados: `python -m pip install --upgrade build twine`

Passo-a-passo (local)
1. Atualize a versão do pacote:
   - Edite `singular_finance/__init__.py` e atualize `__version__`.
   - Atualize `CHANGELOG.md` com as mudanças.

2. Execute os testes:

```bash
pytest -q
```

3. Gere as distribuições:

```bash
python -m build
```

Isso cria `dist/` com `.tar.gz` e `.whl`.

4. Valide os artefatos:

```bash
twine check dist/*
```

5. Envie para TestPyPI (recomendado antes do PyPI):

```bash
TWINE_USERNAME=__token__ TWINE_PASSWORD=<test-pypi-token> \
  twine upload --repository testpypi dist/*
```

6. Instale o pacote do TestPyPI para checar:

```bash
pip install --index-url https://test.pypi.org/simple/ singular-finance
```

7. Quando estiver satisfeito, publique no PyPI:

```bash
TWINE_USERNAME=__token__ TWINE_PASSWORD=<pypi-token> twine upload dist/*
```

Uso de tokens e CI
- Nunca exponha credenciais no repositório. Use secrets do GitHub Actions: `PYPI_API_TOKEN`.
- No GitHub Actions use `pypa/gh-action-pypi-publish` ou `twine` manualmente com o token.

Exemplo rápido de comando em CI (shell):

```bash
python -m pip install --upgrade build twine
python -m build
python -m pip install twine
# Usa token salvo em ${{ secrets.PYPI_API_TOKEN }}
TWINE_USERNAME=__token__ TWINE_PASSWORD="$PYPI_API_TOKEN" twine upload dist/*
```

Checklist antes de publicar
- [ ] Todos os testes passam (`pytest -q`).
- [ ] Versão atualizada.
- [ ] CHANGELOG atualizado.
- [ ] `pyproject.toml` / `setup.py` com metadados corretos (author, license, classifiers).
- [ ] Licença compatível e arquivo `LICENSE` presente.

Observações
- Este repositório usa `setup.py`. É possível migrar para `pyproject.toml` com [PEP 621] para metadados modernos.
- Considere publicar primeiro em TestPyPI para validar o processo.

