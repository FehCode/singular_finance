# Documentação - Singular Finance

Bem-vindo à documentação mínima do pacote Singular Finance. Esta pasta contém exemplos de uso e referências rápidas.

## Instalação

Instale a versão localmente para desenvolvimento:

```bash
pip install -e .[dev]
```

## Exemplos rápidos

Importar módulos principais:

```python
from singular_finance import FinancialIndicators, ValuationModels, CashFlowAnalysis
import pandas as pd

# construir um DataFrame de exemplo e calcular ROE
df = pd.DataFrame({
    'receita_liquida': [1000000, 1100000, 1200000],
    'lucro_liquido': [100000, 120000, 140000],
    'ativo_total': [2000000, 2200000, 2400000],
    'patrimonio_liquido': [800000, 900000, 1000000],
})

fi = FinancialIndicators(df)
print(fi.calculate_roe())
```

## Estrutura da API (resumo)

- `singular_finance.indicators.FinancialIndicators` — cálculos de KPIs fundamentalistas
- `singular_finance.valuation.ValuationModels` — DCF, múltiplos e valuation
- `singular_finance.cash_flow.CashFlowAnalysis` — análise do fluxo de caixa
- `singular_finance.data.DataCollector` e `DataProcessor` — coleta e limpeza de dados

Para mais exemplos, veja `examples/exemplo_completo.py`.
