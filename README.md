# Singular Finance - Biblioteca de Análise Financeira Corporativa

Singular Finance é uma biblioteca Python completa para análise financeira corporativa, oferecendo ferramentas avançadas para **valuation**, **indicadores financeiros**, **análise fundamentalista**, **análise técnica de ações**, fluxo de caixa, visualização e modelos matemáticos.

## 🚀 Principais recursos

- **Indicadores Financeiros:** ROE, ROA, margens, liquidez, endividamento, eficiência
- **Valuation:** Fluxo de caixa descontado (DCF), múltiplos comparáveis, valuation de ativos
- **Análise Fundamentalista:**
  - Margem bruta, margem operacional, payout, dividend yield
  - Crescimento de receita e lucro, qualidade dos lucros
  - Dívida líquida/EBITDA, ROIC, EBITDA/Receita, eficiência e mais
- **Análise Técnica**:
  - RSI (índice de força relativa), MACD, médias móveis (SMA/EMA)
  - Detecção de padrões de candles (martelo, estrela cadente...)
  - Suporte total para estratégias quantitativas
- **Fluxo de Caixa:** Métricas completas, qualidade do fluxo e análise executiva
- **Visualização:** Gráficos, dashboards, heatmaps e mais
- **Modelos Matemáticos:** Black-Scholes, Monte Carlo, VaR, GARCH, machine learning
- **Coleta de Dados:** Yahoo Finance, automatização de downloads
- **Utilitários Financeiros:** Formatação, cálculos auxiliares, validação

---

## 📦 Instalação

```bash
pip install singular-finance
```

## 🔧 Instalação para Desenvolvimento

```bash
git clone https://github.com/singular-finance/singular-finance.git
cd singular-finance
pip install -e .[dev]
```

---

## 🎯 Exemplo Rápido de Uso

```python
import pandas as pd
from singular_finance import indicators, models

# Dados de exemplo
df = pd.DataFrame({
    "receita_liquida": [100000, 120000],
    "lucro_bruto": [40000, 50000],
    "lucro_liquido": [20000, 25000],
    "ebit": [30000, 35000],
    "ebitda": [31000, 36000],
    "patrimonio_liquido": [100000, 110000],
    "dividendos": [5000, 6000],
    "acoes_circulantes": [10000, 10000]
})

# 📊 Indicadores fundamentalistas
fi = indicators.FinancialIndicators(df)
print("ROE:", fi.calculate_roe())
print("Margem Bruta:", fi.calculate_margem_bruta())
print("Payout:", fi.calculate_payout())
print("Dividend Yield:", fi.calculate_dividend_yield(preco_acao=20.00))

# 📈 Análise técnica
import numpy as np
precos = pd.Series(np.random.uniform(10, 30, 100))
print("RSI:", models.calculate_rsi(precos).tail())
print("MACD:", models.calculate_macd(precos).tail())
```

---

## 📚 Documentação de Indicadores Avançados

A Classe **FinancialIndicators** permite calcular KPIs profissionais:

- **Margem Bruta:** `calculate_margem_bruta()`
- **Margem Operacional (EBIT):** `calculate_margem_operacional()`
- **Payout:** `calculate_payout()`
- **Dividend Yield:** `calculate_dividend_yield(preco_acao)`
- **Crescimento Receita/Lucro:** `calculate_crescimento_receita()`
- **Qualidade dos Lucros:** `calculate_qualidade_lucros()`
- **Dívida Líquida / EBITDA:** `calculate_divida_liquida_ebitda()`
- e muito mais!

**Exemplo:**
```python
fi = indicators.FinancialIndicators(df)
print("Margem Operacional:", fi.calculate_margem_operacional())
print("Crescimento Receita:", fi.calculate_crescimento_receita())
```

---

## 📉 Recursos de Análise Técnica

Funções diretas no módulo `models` para indicadores clássicos:
- **RSI:** `models.calculate_rsi(prices, window=14)`
- **MACD:** `models.calculate_macd(prices)`
- **Médias móveis:** `models.calculate_sma(prices, window)` e `models.calculate_ema(prices, window)`
- **Candles:** `models.is_hammer(open, high, low, close)`

**Exemplo:**
```python
import pandas as pd
from singular_finance import models
precos = pd.Series([...])
rsi = models.calculate_rsi(precos)
macd = models.calculate_macd(precos)
```

---

## 📈 Visualizações

Monte gráficos automáticos com o módulo `visualization`. Exemplo:
```python
from singular_finance import visualization
fig = visualization.plot_financial_metrics(indicadores)
```

---

## 🧪 Testes automatizados

```bash
pytest tests/ -v
```

---

## 🤝 Contribuição
Veja o arquivo [CONTRIBUTING.md](CONTRIBUTING.md).
