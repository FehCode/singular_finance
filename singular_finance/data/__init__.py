"""
Módulo de Coleta e Processamento de Dados

Este módulo contém funções para coleta de dados financeiros de diversas
fontes e processamento de dados para análise.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from typing import Any, Dict, List, Optional, Union, Tuple, cast
import warnings
from datetime import datetime, timedelta
import time


class DataCollector:
    """
    Classe para coleta de dados financeiros de diversas fontes.
    """
    
    def __init__(self):
        """Inicializa o coletor de dados."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_yahoo_finance_data(
        self,
        symbol: str,
        period: str = "5y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Coleta dados do Yahoo Finance.
        
        Args:
            symbol: Símbolo da ação (ex: 'PETR4.SA')
            period: Período dos dados ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Intervalo dos dados ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            DataFrame com dados históricos
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"Nenhum dado encontrado para {symbol}")
            
            # Renomear colunas para português quando presentes
            col_map = {
                'Open': 'Abertura',
                'High': 'Máxima',
                'Low': 'Mínima',
                'Close': 'Fechamento',
                'Adj Close': 'Fechamento Ajustado',
                'Volume': 'Volume',
                'Dividends': 'Dividendos',
                'Stock Splits': 'Split'
            }

            # Aplicar apenas as colunas que existem no DataFrame retornado
            existing_map = {k: v for k, v in col_map.items() if k in data.columns}
            if existing_map:
                data = data.rename(columns=existing_map)

            # o objeto retornado por yfinance pode ser tipado dinamicamente;
            # forçamos o tipo para pd.DataFrame para satisfazer a checagem estática
            return cast(pd.DataFrame, data)
            
        except Exception as e:
            raise Exception(f"Erro ao coletar dados do Yahoo Finance: {str(e)}")
    
    def get_financial_statements(
        self,
        symbol: str,
        statement_type: str = "income"
    ) -> pd.DataFrame:
        """
        Coleta demonstrações financeiras do Yahoo Finance.
        
        Args:
            symbol: Símbolo da ação
            statement_type: Tipo de demonstração ('income', 'balance', 'cashflow')
            
        Returns:
            DataFrame com demonstração financeira
        """
        try:
            ticker = yf.Ticker(symbol)

            if statement_type == "income":
                data = ticker.financials
            elif statement_type == "balance":
                data = ticker.balance_sheet
            elif statement_type == "cashflow":
                data = ticker.cashflow
            else:
                raise ValueError("statement_type deve ser 'income', 'balance' ou 'cashflow'")

            if getattr(data, "empty", False):
                raise ValueError(f"Nenhuma demonstração encontrada para {symbol}")

            return cast(pd.DataFrame, data)
            
        except Exception as e:
            raise Exception(f"Erro ao coletar demonstrações financeiras: {str(e)}")
    
    def get_company_info(self, symbol: str) -> Dict[str, str]:
        """
        Coleta informações da empresa.
        
        Args:
            symbol: Símbolo da ação
            
        Returns:
            Dicionário com informações da empresa
        """
        try:
            ticker = yf.Ticker(symbol)
            info: Any = ticker.info

            # Filtrar informações relevantes
            relevant_info: Dict[str, str] = {
                'nome': str(info.get('longName', '')),
                'setor': str(info.get('sector', '')),
                'industria': str(info.get('industry', '')),
                'pais': str(info.get('country', '')),
                'funcionarios': str(info.get('fullTimeEmployees', 0)),
                'website': str(info.get('website', '')),
                'descricao': str(info.get('longBusinessSummary', '')),
                'moeda': str(info.get('currency', '')),
                'mercado': str(info.get('exchange', ''))
            }

            return relevant_info
            
        except Exception as e:
            raise Exception(f"Erro ao coletar informações da empresa: {str(e)}")
    
    def get_dividend_data(self, symbol: str) -> pd.DataFrame:
        """
        Coleta dados de dividendos.
        
        Args:
            symbol: Símbolo da ação
            
        Returns:
            DataFrame com dados de dividendos
        """
        try:
            ticker = yf.Ticker(symbol)
            dividends = ticker.dividends

            if getattr(dividends, "empty", False):
                return pd.DataFrame()

            return cast(pd.DataFrame, dividends.to_frame('Dividendos'))
            
        except Exception as e:
            raise Exception(f"Erro ao coletar dados de dividendos: {str(e)}")


class DataProcessor:
    """
    Classe para processamento e limpeza de dados financeiros.
    """
    
    def __init__(self):
        """Inicializa o processador de dados."""
        return None
    
    def clean_financial_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Limpa dados financeiros removendo valores inválidos.
        
        Args:
            data: DataFrame com dados financeiros
            
        Returns:
            DataFrame limpo
        """
        cleaned_data = data.copy()
        
        # Remover colunas completamente vazias
        cleaned_data = cleaned_data.dropna(axis=1, how='all')
        
        # Remover linhas completamente vazias
        cleaned_data = cleaned_data.dropna(axis=0, how='all')
        
        # Substituir valores infinitos por NaN
        cleaned_data = cleaned_data.replace([np.inf, -np.inf], np.nan)
        
        # Remover outliers extremos (valores > 10x o desvio padrão)
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if cleaned_data[col].notna().sum() > 0:
                mean = cleaned_data[col].mean()
                std = cleaned_data[col].std()
                if std > 0:
                    cleaned_data[col] = cleaned_data[col].where(
                        abs(cleaned_data[col] - mean) <= 10 * std
                    )
        
        return cleaned_data
    
    def calculate_returns(self, prices: pd.Series, method: str = 'simple') -> pd.Series:
        """
        Calcula retornos dos preços.
        
        Args:
            prices: Série com preços
            method: Método de cálculo ('simple' ou 'log')
            
        Returns:
            Série com retornos
        """
        if method == 'simple':
            returns = prices.pct_change()
        elif method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            raise ValueError("Método deve ser 'simple' ou 'log'")
        
        return returns
    
    def calculate_volatility(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """
        Calcula volatilidade móvel.
        
        Args:
            returns: Série com retornos
            window: Janela de cálculo (padrão 252 dias úteis)
            
        Returns:
            Série com volatilidade anualizada
        """
        volatility = returns.rolling(window=window).std() * np.sqrt(252)
        return volatility
    
    def calculate_moving_averages(
        self,
        prices: pd.Series,
        windows: List[int] = [20, 50, 200]
    ) -> pd.DataFrame:
        """
        Calcula médias móveis.
        
        Args:
            prices: Série com preços
            windows: Lista com janelas das médias móveis
            
        Returns:
            DataFrame com médias móveis
        """
        ma_data = pd.DataFrame(index=prices.index)
        ma_data['Preco'] = prices
        
        for window in windows:
            ma_data[f'MA_{window}'] = prices.rolling(window=window).mean()
        
        return ma_data
    
    def detect_missing_data(self, data: pd.DataFrame) -> Dict[str, Union[int, List]]:
        """
        Detecta dados ausentes no DataFrame.
        
        Args:
            data: DataFrame para análise
            
        Returns:
            Dicionário com informações sobre dados ausentes
        """
        missing_data = {}
        
        # Contar valores ausentes por coluna
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data)) * 100
        
        missing_data['total_missing'] = missing_counts.sum()
        missing_data['columns_with_missing'] = missing_counts[missing_counts > 0].to_dict()
        missing_data['missing_percentages'] = missing_percentages[missing_percentages > 0].to_dict()
        
        # Identificar padrões de dados ausentes
        missing_data['completely_missing_columns'] = missing_counts[missing_counts == len(data)].index.tolist()
        missing_data['partially_missing_columns'] = missing_counts[(missing_counts > 0) & (missing_counts < len(data))].index.tolist()
        
        return missing_data
    
    def fill_missing_data(
        self,
        data: pd.DataFrame,
        method: str = 'forward'
    ) -> pd.DataFrame:
        """
        Preenche dados ausentes.
        
        Args:
            data: DataFrame com dados ausentes
            method: Método de preenchimento ('forward', 'backward', 'mean', 'median', 'interpolate')
            
        Returns:
            DataFrame com dados preenchidos
        """
        filled_data = data.copy()
        
        if method == 'forward':
            filled_data = filled_data.fillna(method='ffill')
        elif method == 'backward':
            filled_data = filled_data.fillna(method='bfill')
        elif method == 'mean':
            numeric_columns = filled_data.select_dtypes(include=[np.number]).columns
            filled_data[numeric_columns] = filled_data[numeric_columns].fillna(filled_data[numeric_columns].mean())
        elif method == 'median':
            numeric_columns = filled_data.select_dtypes(include=[np.number]).columns
            filled_data[numeric_columns] = filled_data[numeric_columns].fillna(filled_data[numeric_columns].median())
        elif method == 'interpolate':
            filled_data = filled_data.interpolate()
        else:
            raise ValueError("Método deve ser 'forward', 'backward', 'mean', 'median' ou 'interpolate'")
        
        return filled_data


def get_stock_data(symbol: str, period: str = "5y") -> pd.DataFrame:
    """
    Função utilitária para coletar dados de ações.
    
    Args:
        symbol: Símbolo da ação
        period: Período dos dados
        
    Returns:
        DataFrame com dados históricos
    """
    collector = DataCollector()
    return collector.get_yahoo_finance_data(symbol, period)


def get_financial_statements(symbol: str, statement_type: str = "income") -> pd.DataFrame:
    """
    Função utilitária para coletar demonstrações financeiras.
    
    Args:
        symbol: Símbolo da ação
        statement_type: Tipo de demonstração
        
    Returns:
        DataFrame com demonstração financeira
    """
    collector = DataCollector()
    return collector.get_financial_statements(symbol, statement_type)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Função utilitária para limpeza de dados.
    
    Args:
        data: DataFrame com dados financeiros
        
    Returns:
        DataFrame limpo
    """
    processor = DataProcessor()
    return processor.clean_financial_data(data)
