"""
Módulo de Utilitários

Este módulo contém funções utilitárias para análise financeira,
incluindo cálculos auxiliares, formatação e validações.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings
from datetime import datetime, timedelta
import re


class FinancialUtils:
    """
    Classe com utilitários para análise financeira.
    """
    
    def __init__(self):
        """Inicializa a classe de utilitários."""
        pass
    
    def format_currency(
        self,
        value: Union[float, int],
        currency: str = "BRL",
        decimals: int = 2
    ) -> str:
        """
        Formata valores monetários.
        
        Args:
            value: Valor a ser formatado
            currency: Moeda ('BRL', 'USD', 'EUR')
            decimals: Número de casas decimais
            
        Returns:
            String formatada
        """
        if pd.isna(value) or value is None:
            return "N/A"
        
        # Símbolos das moedas
        currency_symbols = {
            'BRL': 'R$',
            'USD': '$',
            'EUR': '€',
            'GBP': '£'
        }
        
        symbol = currency_symbols.get(currency, currency)
        
        # Formatar número
        if abs(value) >= 1e12:
            formatted_value = f"{value/1e12:.{decimals}f}T"
        elif abs(value) >= 1e9:
            formatted_value = f"{value/1e9:.{decimals}f}B"
        elif abs(value) >= 1e6:
            formatted_value = f"{value/1e6:.{decimals}f}M"
        elif abs(value) >= 1e3:
            formatted_value = f"{value/1e3:.{decimals}f}K"
        else:
            formatted_value = f"{value:.{decimals}f}"
        
        return f"{symbol} {formatted_value}"
    
    def format_percentage(
        self,
        value: Union[float, int],
        decimals: int = 2,
        show_sign: bool = True
    ) -> str:
        """
        Formata valores percentuais.
        
        Args:
            value: Valor a ser formatado (em decimal)
            decimals: Número de casas decimais
            show_sign: Se deve mostrar sinal
            
        Returns:
            String formatada
        """
        if pd.isna(value) or value is None:
            return "N/A"
        
        percentage = value * 100
        
        if show_sign and percentage > 0:
            return f"+{percentage:.{decimals}f}%"
        else:
            return f"{percentage:.{decimals}f}%"
    
    def calculate_cagr(
        self,
        initial_value: float,
        final_value: float,
        years: int
    ) -> float:
        """
        Calcula CAGR (Compound Annual Growth Rate).
        
        Args:
            initial_value: Valor inicial
            final_value: Valor final
            years: Número de anos
            
        Returns:
            CAGR em decimal
        """
        if initial_value <= 0 or years <= 0:
            return 0.0
        
        return (final_value / initial_value) ** (1 / years) - 1
    
    def calculate_npv(
        self,
        cash_flows: List[float],
        discount_rate: float,
        initial_investment: float = 0
    ) -> float:
        """
        Calcula NPV (Net Present Value).
        
        Args:
            cash_flows: Lista com fluxos de caixa
            discount_rate: Taxa de desconto
            initial_investment: Investimento inicial
            
        Returns:
            NPV
        """
        npv = -initial_investment
        
        for i, cf in enumerate(cash_flows):
            npv += cf / ((1 + discount_rate) ** (i + 1))
        
        return npv
    
    def calculate_irr(
        self,
        cash_flows: List[float],
        initial_guess: float = 0.1
    ) -> float:
        """
        Calcula IRR (Internal Rate of Return).
        
        Args:
            cash_flows: Lista com fluxos de caixa
            initial_guess: Chute inicial para IRR
            
        Returns:
            IRR em decimal
        """
        from scipy.optimize import fsolve
        
        def npv_function(rate):
            return self.calculate_npv(cash_flows, rate)
        
        try:
            irr = fsolve(npv_function, initial_guess)[0]
            return irr
        except:
            return np.nan
    
    def calculate_payback_period(
        self,
        cash_flows: List[float],
        initial_investment: float
    ) -> float:
        """
        Calcula período de payback.
        
        Args:
            cash_flows: Lista com fluxos de caixa
            initial_investment: Investimento inicial
            
        Returns:
            Período de payback em anos
        """
        cumulative_cf = -initial_investment
        
        for i, cf in enumerate(cash_flows):
            cumulative_cf += cf
            if cumulative_cf >= 0:
                # Interpolação linear para período fracionário
                if i == 0:
                    return 0
                else:
                    prev_cumulative = cumulative_cf - cf
                    return i - 1 + abs(prev_cumulative) / cf
        
        return len(cash_flows)  # Nunca recuperou o investimento
    
    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calcula Sharpe Ratio.
        
        Args:
            returns: Série com retornos
            risk_free_rate: Taxa livre de risco
            
        Returns:
            Sharpe Ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        return excess_returns.mean() / returns.std() if returns.std() != 0 else 0.0
    
    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calcula Sortino Ratio.
        
        Args:
            returns: Série com retornos
            risk_free_rate: Taxa livre de risco
            
        Returns:
            Sortino Ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        downside_deviation = downside_returns.std()
        
        return excess_returns.mean() / downside_deviation if downside_deviation != 0 else 0.0
    
    def calculate_calmar_ratio(
        self,
        returns: pd.Series,
        max_drawdown: float
    ) -> float:
        """
        Calcula Calmar Ratio.
        
        Args:
            returns: Série com retornos
            max_drawdown: Máximo drawdown
            
        Returns:
            Calmar Ratio
        """
        if max_drawdown == 0:
            return float('inf') if returns.mean() > 0 else 0.0
        
        annual_return = returns.mean() * 252  # Assumindo dados diários
        return annual_return / abs(max_drawdown)
    
    def calculate_beta(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series
    ) -> float:
        """
        Calcula Beta do ativo em relação ao mercado.
        
        Args:
            asset_returns: Retornos do ativo
            market_returns: Retornos do mercado
            
        Returns:
            Beta
        """
        # Alinhar índices
        aligned_data = pd.DataFrame({
            'asset': asset_returns,
            'market': market_returns
        }).dropna()
        
        if len(aligned_data) < 2:
            return 1.0
        
        covariance = np.cov(aligned_data['asset'], aligned_data['market'])[0, 1]
        market_variance = np.var(aligned_data['market'])
        
        return covariance / market_variance if market_variance != 0 else 1.0
    
    def calculate_treynor_ratio(
        self,
        returns: pd.Series,
        beta: float,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calcula Treynor Ratio.
        
        Args:
            returns: Série com retornos
            beta: Beta do ativo
            risk_free_rate: Taxa livre de risco
            
        Returns:
            Treynor Ratio
        """
        if beta == 0:
            return float('inf') if returns.mean() > risk_free_rate else 0.0
        
        excess_return = returns.mean() - risk_free_rate
        return excess_return / beta
    
    def calculate_jensen_alpha(
        self,
        returns: pd.Series,
        market_returns: pd.Series,
        risk_free_rate: float = 0.02,
        beta: Optional[float] = None
    ) -> float:
        """
        Calcula Jensen's Alpha.
        
        Args:
            returns: Retornos do ativo
            market_returns: Retornos do mercado
            risk_free_rate: Taxa livre de risco
            beta: Beta do ativo (calculado se não fornecido)
            
        Returns:
            Jensen's Alpha
        """
        if beta is None:
            beta = self.calculate_beta(returns, market_returns)
        
        expected_return = risk_free_rate + beta * (market_returns.mean() - risk_free_rate)
        actual_return = returns.mean()
        
        return actual_return - expected_return
    
    def validate_financial_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida dados financeiros.
        
        Args:
            data: DataFrame com dados financeiros
            
        Returns:
            Dicionário com resultados da validação
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        # Verificar se DataFrame não está vazio
        if data.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("DataFrame está vazio")
            return validation_results
        
        # Verificar colunas numéricas
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            validation_results['warnings'].append("Nenhuma coluna numérica encontrada")
        
        # Verificar valores infinitos
        for col in numeric_columns:
            infinite_count = np.isinf(data[col]).sum()
            if infinite_count > 0:
                validation_results['warnings'].append(f"Coluna '{col}' contém {infinite_count} valores infinitos")
        
        # Verificar valores negativos em colunas que não deveriam ter
        non_negative_columns = ['receita_liquida', 'ativo_total', 'patrimonio_liquido']
        for col in non_negative_columns:
            if col in data.columns:
                negative_count = (data[col] < 0).sum()
                if negative_count > 0:
                    validation_results['warnings'].append(f"Coluna '{col}' contém {negative_count} valores negativos")
        
        # Resumo estatístico
        validation_results['summary'] = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'numeric_columns': len(numeric_columns),
            'missing_values': data.isnull().sum().sum(),
            'duplicate_rows': data.duplicated().sum()
        }
        
        return validation_results
    
    def convert_currency(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
        exchange_rate: Optional[float] = None
    ) -> float:
        """
        Converte valores entre moedas.
        
        Args:
            amount: Valor a ser convertido
            from_currency: Moeda origem
            to_currency: Moeda destino
            exchange_rate: Taxa de câmbio (opcional)
            
        Returns:
            Valor convertido
        """
        if from_currency == to_currency:
            return amount
        
        if exchange_rate is None:
            # Taxas de câmbio aproximadas (para demonstração)
            rates = {
                'USD_BRL': 5.0,
                'BRL_USD': 0.2,
                'EUR_BRL': 5.5,
                'BRL_EUR': 0.18,
                'GBP_BRL': 6.2,
                'BRL_GBP': 0.16
            }
            
            rate_key = f"{from_currency}_{to_currency}"
            if rate_key not in rates:
                raise ValueError(f"Taxa de câmbio não disponível para {rate_key}")
            
            exchange_rate = rates[rate_key]
        
        return amount * exchange_rate
    
    def calculate_inflation_adjusted_return(
        self,
        nominal_return: float,
        inflation_rate: float
    ) -> float:
        """
        Calcula retorno ajustado pela inflação.
        
        Args:
            nominal_return: Retorno nominal
            inflation_rate: Taxa de inflação
            
        Returns:
            Retorno real
        """
        return (1 + nominal_return) / (1 + inflation_rate) - 1
    
    def calculate_effective_annual_rate(
        self,
        nominal_rate: float,
        compounding_periods: int
    ) -> float:
        """
        Calcula taxa efetiva anual.
        
        Args:
            nominal_rate: Taxa nominal
            compounding_periods: Número de períodos de capitalização
            
        Returns:
            Taxa efetiva anual
        """
        return (1 + nominal_rate / compounding_periods) ** compounding_periods - 1


def format_currency(value: Union[float, int], currency: str = "BRL") -> str:
    """
    Função utilitária para formatação de moeda.
    
    Args:
        value: Valor a ser formatado
        currency: Moeda
        
    Returns:
        String formatada
    """
    utils = FinancialUtils()
    return utils.format_currency(value, currency)


def format_percentage(value: Union[float, int], decimals: int = 2) -> str:
    """
    Função utilitária para formatação de percentual.
    
    Args:
        value: Valor a ser formatado
        decimals: Número de casas decimais
        
    Returns:
        String formatada
    """
    utils = FinancialUtils()
    return utils.format_percentage(value, decimals)


def calculate_cagr(initial_value: float, final_value: float, years: int) -> float:
    """
    Função utilitária para cálculo de CAGR.
    
    Args:
        initial_value: Valor inicial
        final_value: Valor final
        years: Número de anos
        
    Returns:
        CAGR
    """
    utils = FinancialUtils()
    return utils.calculate_cagr(initial_value, final_value, years)


def calculate_npv(cash_flows: List[float], discount_rate: float, initial_investment: float = 0) -> float:
    """
    Função utilitária para cálculo de NPV.
    
    Args:
        cash_flows: Fluxos de caixa
        discount_rate: Taxa de desconto
        initial_investment: Investimento inicial
        
    Returns:
        NPV
    """
    utils = FinancialUtils()
    return utils.calculate_npv(cash_flows, discount_rate, initial_investment)
