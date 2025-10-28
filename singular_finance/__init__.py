"""
Singular Finance - Biblioteca de Análise Financeira Corporativa

Uma biblioteca Python completa para análise financeira corporativa,
valuation, indicadores financeiros, fluxo de caixa e visualização de dados.
"""

__version__ = "1.0.2"
__author__ = "Singular Finance Team"
__email__ = "contact@singularfinance.com"

# Importações principais dos módulos
from . import indicators
from . import valuation
from . import cash_flow
from . import visualization
from . import models
from . import data
from . import utils

# Importações específicas para facilitar o uso
from .indicators import FinancialIndicators
from .valuation import ValuationModels
from .cash_flow import CashFlowAnalysis
from .visualization import FinancialCharts
from .models import MathematicalModels

__all__ = [
    "indicators",
    "valuation", 
    "cash_flow",
    "visualization",
    "models",
    "data",
    "utils",
    "FinancialIndicators",
    "ValuationModels",
    "CashFlowAnalysis",
    "FinancialCharts",
    "MathematicalModels",
]
