"""
Módulo de Indicadores Financeiros

Este módulo contém classes e funções para cálculo de indicadores financeiros
essenciais para análise corporativa.
"""

__all__ = [
    "FinancialIndicators",
    "calculate_all",
    "calculate_roe",
    "calculate_roa",
    "calculate_margem_liquida",
    "calculate_liquidez_corrente",
]

import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


class FinancialIndicators:
    """
    Classe principal para cálculo de indicadores financeiros.

    Esta classe fornece métodos para calcular diversos indicadores financeiros
    como rentabilidade, liquidez, endividamento e eficiência operacional.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Inicializa a classe com dados financeiros.

        Args:
            data: DataFrame com dados financeiros da empresa
        """
        self.data = data.copy()
        self._validate_data()

    def _validate_data(self) -> None:
        """Valida se os dados contêm as colunas necessárias."""
        required_columns = [
            "receita_liquida",
            "lucro_liquido",
            "ativo_total",
            "patrimonio_liquido",
            "ativo_circulante",
            "passivo_circulante",
        ]

        missing_columns = [
            col for col in required_columns if col not in self.data.columns
        ]
        if missing_columns:
            raise ValueError(f"Colunas ausentes: {missing_columns}")

    def calculate_roe(self) -> float:
        """
        Calcula o Return on Equity (ROE).

        ROE = Lucro Líquido / Patrimônio Líquido

        Returns:
            ROE em percentual
        """
        lucro_liquido = float(self.data["lucro_liquido"].iloc[-1])
        patrimonio_liquido = float(self.data["patrimonio_liquido"].iloc[-1])

        if patrimonio_liquido == 0:
            return None

        return (lucro_liquido / patrimonio_liquido) * 100

    def calculate_roa(self) -> float:
        """
        Calcula o Return on Assets (ROA).

        ROA = Lucro Líquido / Ativo Total

        Returns:
            ROA em percentual
        """
        lucro_liquido = float(self.data["lucro_liquido"].iloc[-1])
        ativo_total = float(self.data["ativo_total"].iloc[-1])

        if ativo_total == 0:
            return None

        return (lucro_liquido / ativo_total) * 100

    def calculate_margem_liquida(self) -> float:
        """
        Calcula a Margem Líquida.

        Margem Líquida = Lucro Líquido / Receita Líquida

        Returns:
            Margem líquida em percentual
        """
        lucro_liquido = float(self.data["lucro_liquido"].iloc[-1])
        receita_liquida = float(self.data["receita_liquida"].iloc[-1])

        if receita_liquida == 0:
            return None

        return (lucro_liquido / receita_liquida) * 100

    def calculate_margem_ebitda(self) -> float:
        """
        Calcula a Margem EBITDA.

        Margem EBITDA = EBITDA / Receita Líquida

        Returns:
            Margem EBITDA em percentual
        """
        ebitda = float(self.data["ebitda"].iloc[-1])
        receita_liquida = float(self.data["receita_liquida"].iloc[-1])

        if receita_liquida == 0:
            return None

        return (ebitda / receita_liquida) * 100

    def calculate_liquidez_corrente(self) -> float:
        """
        Calcula a Liquidez Corrente.

        Liquidez Corrente = Ativo Circulante / Passivo Circulante

        Returns:
            Liquidez corrente
        """
        ativo_circulante = float(self.data["ativo_circulante"].iloc[-1])
        passivo_circulante = float(self.data["passivo_circulante"].iloc[-1])

        if passivo_circulante == 0:
            return None

        return ativo_circulante / passivo_circulante

    def calculate_liquidez_seca(self) -> float:
        """
        Calcula a Liquidez Seca.

        Liquidez Seca = (Ativo Circulante - Estoque) / Passivo Circulante

        Returns:
            Liquidez seca
        """
        ativo_circulante = float(self.data["ativo_circulante"].iloc[-1])
        passivo_circulante = float(self.data["passivo_circulante"].iloc[-1])

        if "estoque" in self.data.columns:
            estoque = float(self.data["estoque"].iloc[-1])
        else:
            estoque = 0

        if passivo_circulante == 0:
            return None

        return (ativo_circulante - estoque) / passivo_circulante

    def calculate_endividamento_total(self) -> float:
        """
        Calcula o Endividamento Total.

        Endividamento Total = Passivo Total / Ativo Total

        Returns:
            Endividamento total em percentual
        """
        passivo_total = float(self.data["passivo_total"].iloc[-1])
        ativo_total = float(self.data["ativo_total"].iloc[-1])

        if ativo_total == 0:
            return None

        return (passivo_total / ativo_total) * 100

    def calculate_giro_ativo(self) -> float:
        """
        Calcula o Giro do Ativo.

        Giro do Ativo = Receita Líquida / Ativo Total

        Returns:
            Giro do ativo
        """
        receita_liquida = float(self.data["receita_liquida"].iloc[-1])
        ativo_total = float(self.data["ativo_total"].iloc[-1])

        if ativo_total == 0:
            return None

        return receita_liquida / ativo_total

    def calculate_pe_ratio(self, preco_acao: float) -> Optional[float]:
        """
        Calcula o P/E Ratio (Price-to-Earnings).

        P/E = Preço da Ação / Lucro por Ação

        Args:
            preco_acao: Preço atual da ação

        Returns:
            P/E ratio
        """
        lucro_liquido = float(self.data["lucro_liquido"].iloc[-1])
        acoes_circulantes = float(self.data["acoes_circulantes"].iloc[-1])

        if acoes_circulantes == 0:
            return None

        lucro_por_acao = lucro_liquido / acoes_circulantes

        if lucro_por_acao == 0:
            return None

        return preco_acao / lucro_por_acao

    def calculate_pb_ratio(self, preco_acao: float) -> Optional[float]:
        """
        Calcula o P/B Ratio (Price-to-Book).

        P/B = Preço da Ação / Valor Contábil por Ação

        Args:
            preco_acao: Preço atual da ação

        Returns:
            P/B ratio
        """
        patrimonio_liquido = float(self.data["patrimonio_liquido"].iloc[-1])
        acoes_circulantes = float(self.data["acoes_circulantes"].iloc[-1])

        if acoes_circulantes == 0:
            return None

        valor_contabil_por_acao = patrimonio_liquido / acoes_circulantes

        if valor_contabil_por_acao == 0:
            return None

        return preco_acao / valor_contabil_por_acao

    def _calculate_indicator(self, func, *args) -> Optional[float]:
        """Helper function to calculate an indicator and handle exceptions."""
        try:
            return func(*args)
        except (ValueError, ZeroDivisionError):
            return None

    def calculate_all_indicators(
        self, preco_acao: Optional[float] = None
    ) -> Dict[str, Optional[float]]:
        """
        Calcula todos os indicadores financeiros disponíveis.

        Args:
            preco_acao: Preço da ação para cálculo de múltiplos (opcional)

        Returns:
            Dicionário com todos os indicadores calculados
        """
        indicators = {
            "roe": self._calculate_indicator(self.calculate_roe),
            "roa": self._calculate_indicator(self.calculate_roa),
            "margem_liquida": self._calculate_indicator(self.calculate_margem_liquida),
            "margem_ebitda": self._calculate_indicator(self.calculate_margem_ebitda),
            "liquidez_corrente": self._calculate_indicator(self.calculate_liquidez_corrente),
            "liquidez_seca": self._calculate_indicator(self.calculate_liquidez_seca),
            "endividamento_total": self._calculate_indicator(self.calculate_endividamento_total),
            "giro_ativo": self._calculate_indicator(self.calculate_giro_ativo),
        }

        if preco_acao is not None:
            indicators["pe_ratio"] = self._calculate_indicator(self.calculate_pe_ratio, preco_acao)
            indicators["pb_ratio"] = self._calculate_indicator(self.calculate_pb_ratio, preco_acao)

        return indicators

    def calculate_margem_bruta(self) -> float:
        """
        Calcula a Margem Bruta.
        Margem Bruta = Lucro Bruto / Receita Líquida
        Returns:
            Margem bruta em percentual
        """
        lucro_bruto = self.data["lucro_bruto"].iloc[-1]
        receita_liquida = self.data["receita_liquida"].iloc[-1]
        if receita_liquida == 0:
            return None
        return (lucro_bruto / receita_liquida) * 100

    def calculate_margem_operacional(self) -> float:
        """
        Calcula a Margem Operacional.
        Margem Operacional = EBIT / Receita Líquida
        Returns:
            Margem operacional em percentual
        """
        ebit = self.data["ebit"].iloc[-1]
        receita_liquida = self.data["receita_liquida"].iloc[-1]
        if receita_liquida == 0:
            return None
        return (ebit / receita_liquida) * 100

    def calculate_payout(self) -> float:
        """
        Calcula o Payout.
        Payout = Dividendos / Lucro Líquido
        Returns:
            Payout em percentual
        """
        dividendos = self.data["dividendos"].iloc[-1]
        lucro_liquido = self.data["lucro_liquido"].iloc[-1]
        if lucro_liquido == 0:
            return None
        return (dividendos / lucro_liquido) * 100

    def calculate_dividend_yield(self, preco_acao: float) -> float:
        """
        Calcula o Dividend Yield.
        Dividend Yield = Dividendos por Ação / Preço da Ação
        Returns:
            Dividend yield em percentual
        """
        dividendos = self.data["dividendos"].iloc[-1]
        acoes = self.data["acoes_circulantes"].iloc[-1]
        if acoes == 0 or preco_acao == 0:
            return None
        dy = (dividendos / acoes) / preco_acao
        return dy * 100

    def calculate_crescimento_receita(self) -> float:
        """
        Calcula o crescimento percentual da receita líquida ano a ano.
        Returns:
            Crescimento em percentual
        """
        r = self.data["receita_liquida"]
        if r.iloc[-2] == 0:
            return None
        return ((r.iloc[-1] / r.iloc[-2]) - 1) * 100

    def calculate_crescimento_lucro(self) -> float:
        """
        Calcula o crescimento do lucro líquido ano a ano.
        Returns:
            Crescimento em percentual
        """
        l = self.data["lucro_liquido"]
        if l.iloc[-2] == 0:
            return None
        return ((l.iloc[-1] / l.iloc[-2]) - 1) * 100

    def calculate_roe_cagr(self, anos: int = 3) -> float:
        """
        Calcula o CAGR do ROE nos últimos anos.
        """
        rs = self.data["lucro_liquido"].tail(anos) / self.data[
            "patrimonio_liquido"
        ].tail(anos)
        if len(rs) < 2 or rs.iloc[0] == 0:
            return None
        return ((rs.iloc[-1] / rs.iloc[0]) ** (1 / (len(rs) - 1)) - 1) * 100

    def calculate_ebitda_atividade(self) -> float:
        """
        Calcula EBITDA / Receita Líquida (ou margem EBITDA).
        """
        e = self.data["ebitda"].iloc[-1]
        r = self.data["receita_liquida"].iloc[-1]
        if r == 0:
            return None
        return (e / r) * 100

    def calculate_divida_liquida_ebitda(self) -> float:
        """
        Calcula Dívida Líquida / EBITDA
        """
        d = self.data["divida_liquida"].iloc[-1]
        e = self.data["ebitda"].iloc[-1]
        if e == 0:
            return None
        return d / e

    def calculate_qualidade_lucros(self) -> float:
        """
        Proporção Fluxo de Caixa Operacional / Lucro Líquido.
        """
        fco = self.data["fluxo_caixa_operacional"].iloc[-1]
        ll = self.data["lucro_liquido"].iloc[-1]
        if ll == 0:
            return None
        return fco / ll

def calculate_all(
    data: pd.DataFrame, preco_acao: Optional[float] = None
) -> Dict[str, Optional[float]]:
    """
    Função utilitária para calcular todos os indicadores financeiros.

    Args:
        data: DataFrame com dados financeiros
        preco_acao: Preço da ação para cálculo de múltiplos (opcional)

    Returns:
        Dicionário com todos os indicadores calculados
    """
    calculator = FinancialIndicators(data)
    return calculator.calculate_all_indicators(preco_acao)


def calculate_roe(data: pd.DataFrame) -> float:
    """Calcula ROE para dados fornecidos."""
    calculator = FinancialIndicators(data)
    return calculator.calculate_roe()


def calculate_roa(data: pd.DataFrame) -> float:
    """Calcula ROA para dados fornecidos."""
    calculator = FinancialIndicators(data)
    return calculator.calculate_roa()


def calculate_margem_liquida(data: pd.DataFrame) -> float:
    """Calcula Margem Líquida para dados fornecidos."""
    calculator = FinancialIndicators(data)
    return calculator.calculate_margem_liquida()


def calculate_liquidez_corrente(data: pd.DataFrame) -> float:
    """Calcula Liquidez Corrente para dados fornecidos."""
    calculator = FinancialIndicators(data)
    return calculator.calculate_liquidez_corrente()
 
