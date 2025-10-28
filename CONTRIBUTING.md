# Singular Finance - Guia de Contribuição

Obrigado por considerar contribuir para o Singular Finance! Este documento fornece diretrizes para contribuições eficazes.

## 🚀 Como Contribuir

### 1. Configuração do Ambiente de Desenvolvimento

```bash
# Clone o repositório
git clone https://github.com/singular-finance/singular-finance.git
cd singular-finance

# Instale em modo de desenvolvimento
pip install -e .

# Instale dependências de desenvolvimento
pip install -e ".[dev]"
```

### 2. Estrutura do Projeto

```
singular_finance/
├── singular_finance/          # Código principal da biblioteca
│   ├── indicators/            # Indicadores financeiros
│   ├── valuation/            # Modelos de valuation
│   ├── cash_flow/            # Análise de fluxo de caixa
│   ├── visualization/        # Visualização de dados
│   ├── models/               # Modelos matemáticos
│   ├── data/                 # Coleta e processamento de dados
│   └── utils/                # Utilitários
├── tests/                    # Testes unitários
├── examples/                 # Exemplos de uso
├── docs/                     # Documentação
└── setup.py                  # Configuração do pacote
```

### 3. Padrões de Código

#### Python Style Guide
- Siga o PEP 8
- Use type hints quando possível
- Documente funções e classes com docstrings
- Mantenha linhas com máximo de 88 caracteres

#### Estrutura de Commits
Use o padrão Conventional Commits:
```
feat: adiciona nova funcionalidade
fix: corrige bug
docs: atualiza documentação
test: adiciona testes
refactor: refatora código
```

### 4. Processo de Contribuição

1. **Fork** o repositório
2. **Crie** uma branch para sua feature (`git checkout -b feature/nova-funcionalidade`)
3. **Commit** suas mudanças (`git commit -m 'feat: adiciona nova funcionalidade'`)
4. **Push** para a branch (`git push origin feature/nova-funcionalidade`)
5. **Abra** um Pull Request

### 5. Testes

Execute os testes antes de submeter:

```bash
# Executar todos os testes
python -m pytest tests/

# Executar com cobertura
python -m pytest tests/ --cov=singular_finance

# Executar testes específicos
python -m pytest tests/test_indicators.py
```

### 6. Documentação

- Atualize a documentação para novas funcionalidades
- Adicione exemplos de uso quando apropriado
- Mantenha o README.md atualizado

### 7. Tipos de Contribuições

#### 🐛 Reportar Bugs
- Use o template de issue para bugs
- Inclua informações sobre ambiente e reprodução

#### ✨ Sugerir Funcionalidades
- Use o template de issue para features
- Descreva o caso de uso e benefícios

#### 💻 Contribuir com Código
- Implemente funcionalidades seguindo os padrões
- Adicione testes para novo código
- Atualize documentação

#### 📚 Melhorar Documentação
- Corrija erros de documentação
- Adicione exemplos mais claros
- Traduza para outros idiomas

### 8. Checklist para Pull Requests

- [ ] Código segue os padrões de estilo
- [ ] Testes passam
- [ ] Documentação atualizada
- [ ] Exemplos funcionam
- [ ] Commits seguem convenção
- [ ] Branch atualizada com main

### 9. Comunidade

- **Discord**: [Link do servidor]
- **Email**: dev@singularfinance.com
- **Issues**: Use GitHub Issues para discussões

### 10. Licença

Ao contribuir, você concorda que suas contribuições serão licenciadas sob a Licença MIT.

---

Obrigado por contribuir para o Singular Finance! 🎉
