"""
Módulo de Indicadores Financeiros

Este módulo contém classes e funções para cálculo de indicadores financeiros
essenciais para análise corporativa.
"""

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

    def _validate_data(self):
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
            warnings.warn(f"Colunas ausentes: {missing_columns}")

    def calculate_roe(self) -> float:
        """
        Calcula o Return on Equity (ROE).

        ROE = Lucro Líquido / Patrimônio Líquido

        Returns:
            ROE em percentual
        """
        if (
            "lucro_liquido" not in self.data.columns
            or "patrimonio_liquido" not in self.data.columns
        ):
            raise ValueError(
                "Colunas 'lucro_liquido' e 'patrimonio_liquido' são necessárias"
            )

        lucro_liquido = self.data["lucro_liquido"].iloc[-1]
        patrimonio_liquido = self.data["patrimonio_liquido"].iloc[-1]

        if patrimonio_liquido == 0:
            return 0.0

        return (lucro_liquido / patrimonio_liquido) * 100

    def calculate_roa(self) -> float:
        """
        Calcula o Return on Assets (ROA).

        ROA = Lucro Líquido / Ativo Total

        Returns:
            ROA em percentual
        """
        if (
            "lucro_liquido" not in self.data.columns
            or "ativo_total" not in self.data.columns
        ):
            raise ValueError("Colunas 'lucro_liquido' e 'ativo_total' são necessárias")

        lucro_liquido = self.data["lucro_liquido"].iloc[-1]
        ativo_total = self.data["ativo_total"].iloc[-1]

        if ativo_total == 0:
            return 0.0

        return (lucro_liquido / ativo_total) * 100

    def calculate_margem_liquida(self) -> float:
        """
        Calcula a Margem Líquida.

        Margem Líquida = Lucro Líquido / Receita Líquida

        Returns:
            Margem líquida em percentual
        """
        if (
            "lucro_liquido" not in self.data.columns
            or "receita_liquida" not in self.data.columns
        ):
            raise ValueError(
                "Colunas 'lucro_liquido' e 'receita_liquida' são necessárias"
            )

        lucro_liquido = self.data["lucro_liquido"].iloc[-1]
        receita_liquida = self.data["receita_liquida"].iloc[-1]

        if receita_liquida == 0:
            return 0.0

        return (lucro_liquido / receita_liquida) * 100

    def calculate_margem_ebitda(self) -> float:
        """
        Calcula a Margem EBITDA.

        Margem EBITDA = EBITDA / Receita Líquida

        Returns:
            Margem EBITDA em percentual
        """
        if (
            "ebitda" not in self.data.columns
            or "receita_liquida" not in self.data.columns
        ):
            raise ValueError("Colunas 'ebitda' e 'receita_liquida' são necessárias")

        ebitda = self.data["ebitda"].iloc[-1]
        receita_liquida = self.data["receita_liquida"].iloc[-1]

        if receita_liquida == 0:
            return 0.0

        return (ebitda / receita_liquida) * 100

    def calculate_liquidez_corrente(self) -> float:
        """
        Calcula a Liquidez Corrente.

        Liquidez Corrente = Ativo Circulante / Passivo Circulante

        Returns:
            Liquidez corrente
        """
        if (
            "ativo_circulante" not in self.data.columns
            or "passivo_circulante" not in self.data.columns
        ):
            raise ValueError(
                "Colunas 'ativo_circulante' e 'passivo_circulante' são necessárias"
            )

        ativo_circulante = self.data["ativo_circulante"].iloc[-1]
        passivo_circulante = self.data["passivo_circulante"].iloc[-1]

        if passivo_circulante == 0:
            return float("inf")

        return ativo_circulante / passivo_circulante

    def calculate_liquidez_seca(self) -> float:
        """
        Calcula a Liquidez Seca.

        Liquidez Seca = (Ativo Circulante - Estoque) / Passivo Circulante

        Returns:
            Liquidez seca
        """
        if (
            "ativo_circulante" not in self.data.columns
            or "passivo_circulante" not in self.data.columns
        ):
            raise ValueError(
                "Colunas 'ativo_circulante' e 'passivo_circulante' são necessárias"
            )

        ativo_circulante = self.data["ativo_circulante"].iloc[-1]
        passivo_circulante = self.data["passivo_circulante"].iloc[-1]

        if "estoque" in self.data.columns:
            estoque = self.data["estoque"].iloc[-1]
        else:
            estoque = 0

        if passivo_circulante == 0:
            return float("inf")

        return (ativo_circulante - estoque) / passivo_circulante

    def calculate_endividamento_total(self) -> float:
        """
        Calcula o Endividamento Total.

        Endividamento Total = Passivo Total / Ativo Total

        Returns:
            Endividamento total em percentual
        """
        if (
            "passivo_total" not in self.data.columns
            or "ativo_total" not in self.data.columns
        ):
            raise ValueError("Colunas 'passivo_total' e 'ativo_total' são necessárias")

        passivo_total = self.data["passivo_total"].iloc[-1]
        ativo_total = self.data["ativo_total"].iloc[-1]

        if ativo_total == 0:
            return 0.0

        return (passivo_total / ativo_total) * 100

    def calculate_giro_ativo(self) -> float:
        """
        Calcula o Giro do Ativo.

        Giro do Ativo = Receita Líquida / Ativo Total

        Returns:
            Giro do ativo
        """
        if (
            "receita_liquida" not in self.data.columns
            or "ativo_total" not in self.data.columns
        ):
            raise ValueError(
                "Colunas 'receita_liquida' e 'ativo_total' são necessárias"
            )

        receita_liquida = self.data["receita_liquida"].iloc[-1]
        ativo_total = self.data["ativo_total"].iloc[-1]

        if ativo_total == 0:
            return 0.0

        return receita_liquida / ativo_total

    def calculate_pe_ratio(self, preco_acao: float) -> float:
        """
        Calcula o P/E Ratio (Price-to-Earnings).

        P/E = Preço da Ação / Lucro por Ação

        Args:
            preco_acao: Preço atual da ação

        Returns:
            P/E ratio
        """
        if (
            "lucro_liquido" not in self.data.columns
            or "acoes_circulantes" not in self.data.columns
        ):
            raise ValueError(
                "Colunas 'lucro_liquido' e 'acoes_circulantes' são necessárias"
            )

        lucro_liquido = self.data["lucro_liquido"].iloc[-1]
        acoes_circulantes = self.data["acoes_circulantes"].iloc[-1]

        if acoes_circulantes == 0:
            return float("inf")

        lucro_por_acao = lucro_liquido / acoes_circulantes

        if lucro_por_acao == 0:
            return float("inf")

        return preco_acao / lucro_por_acao

    def calculate_pb_ratio(self, preco_acao: float) -> float:
        """
        Calcula o P/B Ratio (Price-to-Book).

        P/B = Preço da Ação / Valor Contábil por Ação

        Args:
            preco_acao: Preço atual da ação

        Returns:
            P/B ratio
        """
        if (
            "patrimonio_liquido" not in self.data.columns
            or "acoes_circulantes" not in self.data.columns
        ):
            raise ValueError(
                "Colunas 'patrimonio_liquido' e 'acoes_circulantes' são necessárias"
            )

        patrimonio_liquido = self.data["patrimonio_liquido"].iloc[-1]
        acoes_circulantes = self.data["acoes_circulantes"].iloc[-1]

        if acoes_circulantes == 0:
            return float("inf")

        valor_contabil_por_acao = patrimonio_liquido / acoes_circulantes

        if valor_contabil_por_acao == 0:
            return float("inf")

        return preco_acao / valor_contabil_por_acao

    def calculate_all_indicators(
        self, preco_acao: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calcula todos os indicadores financeiros disponíveis.

        Args:
            preco_acao: Preço da ação para cálculo de múltiplos (opcional)

        Returns:
            Dicionário com todos os indicadores calculados
        """
        indicators = {}

        try:
            indicators["roe"] = self.calculate_roe()
        except (ValueError, ZeroDivisionError):
            indicators["roe"] = None

        try:
            indicators["roa"] = self.calculate_roa()
        except (ValueError, ZeroDivisionError):
            indicators["roa"] = None

        try:
            indicators["margem_liquida"] = self.calculate_margem_liquida()
        except (ValueError, ZeroDivisionError):
            indicators["margem_liquida"] = None

        try:
            indicators["margem_ebitda"] = self.calculate_margem_ebitda()
        except (ValueError, ZeroDivisionError):
            indicators["margem_ebitda"] = None

        try:
            indicators["liquidez_corrente"] = self.calculate_liquidez_corrente()
        except (ValueError, ZeroDivisionError):
            indicators["liquidez_corrente"] = None

        try:
            indicators["liquidez_seca"] = self.calculate_liquidez_seca()
        except (ValueError, ZeroDivisionError):
            indicators["liquidez_seca"] = None

        try:
            indicators["endividamento_total"] = self.calculate_endividamento_total()
        except (ValueError, ZeroDivisionError):
            indicators["endividamento_total"] = None

        try:
            indicators["giro_ativo"] = self.calculate_giro_ativo()
        except (ValueError, ZeroDivisionError):
            indicators["giro_ativo"] = None

        if preco_acao is not None:
            try:
                indicators["pe_ratio"] = self.calculate_pe_ratio(preco_acao)
            except (ValueError, ZeroDivisionError):
                indicators["pe_ratio"] = None

            try:
                indicators["pb_ratio"] = self.calculate_pb_ratio(preco_acao)
            except (ValueError, ZeroDivisionError):
                indicators["pb_ratio"] = None

        return indicators


def calculate_all(
    data: pd.DataFrame, preco_acao: Optional[float] = None
) -> Dict[str, float]:
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

    def calculate_margem_bruta(self) -> float:
        """
        Calcula a Margem Bruta.
        Margem Bruta = Lucro Bruto / Receita Líquida
        Returns:
            Margem bruta em percentual
        """
        if (
            "lucro_bruto" not in self.data.columns
            or "receita_liquida" not in self.data.columns
        ):
            raise ValueError(
                "Colunas 'lucro_bruto' e 'receita_liquida' são necessárias"
            )
        lucro_bruto = self.data["lucro_bruto"].iloc[-1]
        receita_liquida = self.data["receita_liquida"].iloc[-1]
        if receita_liquida == 0:
            return 0.0
        return (lucro_bruto / receita_liquida) * 100

    def calculate_margem_operacional(self) -> float:
        """
        Calcula a Margem Operacional.
        Margem Operacional = EBIT / Receita Líquida
        Returns:
            Margem operacional em percentual
        """
        if (
            "ebit" not in self.data.columns
            or "receita_liquida" not in self.data.columns
        ):
            raise ValueError("Colunas 'ebit' e 'receita_liquida' são necessárias")
        ebit = self.data["ebit"].iloc[-1]
        receita_liquida = self.data["receita_liquida"].iloc[-1]
        if receita_liquida == 0:
            return 0.0
        return (ebit / receita_liquida) * 100

    def calculate_payout(self) -> float:
        """
        Calcula o Payout.
        Payout = Dividendos / Lucro Líquido
        Returns:
            Payout em percentual
        """
        if (
            "dividendos" not in self.data.columns
            or "lucro_liquido" not in self.data.columns
        ):
            raise ValueError("Colunas 'dividendos' e 'lucro_liquido' são necessárias")
        dividendos = self.data["dividendos"].iloc[-1]
        lucro_liquido = self.data["lucro_liquido"].iloc[-1]
        if lucro_liquido == 0:
            return 0.0
        return (dividendos / lucro_liquido) * 100

    def calculate_dividend_yield(self, preco_acao: float) -> float:
        """
        Calcula o Dividend Yield.
        Dividend Yield = Dividendos por Ação / Preço da Ação
        Returns:
            Dividend yield em percentual
        """
        if (
            "dividendos" not in self.data.columns
            or "acoes_circulantes" not in self.data.columns
        ):
            raise ValueError(
                "Colunas 'dividendos' e 'acoes_circulantes' são necessárias"
            )
        dividendos = self.data["dividendos"].iloc[-1]
        acoes = self.data["acoes_circulantes"].iloc[-1]
        if acoes == 0 or preco_acao == 0:
            return 0.0
        dy = (dividendos / acoes) / preco_acao
        return dy * 100

    def calculate_crescimento_receita(self) -> float:
        """
        Calcula o crescimento percentual da receita líquida ano a ano.
        Returns:
            Crescimento em percentual
        """
        if (
            "receita_liquida" not in self.data.columns
            or len(self.data["receita_liquida"]) < 2
        ):
            return 0.0
        r = self.data["receita_liquida"]
        return ((r.iloc[-1] / r.iloc[-2]) - 1) * 100

    def calculate_crescimento_lucro(self) -> float:
        """
        Calcula o crescimento do lucro líquido ano a ano.
        Returns:
            Crescimento em percentual
        """
        if (
            "lucro_liquido" not in self.data.columns
            or len(self.data["lucro_liquido"]) < 2
        ):
            return 0.0
        l = self.data["lucro_liquido"]
        return ((l.iloc[-1] / l.iloc[-2]) - 1) * 100

    def calculate_roe_cagr(self, anos: int = 3) -> float:
        """
        Calcula o CAGR do ROE nos últimos anos.
        """
        if (
            "lucro_liquido" not in self.data.columns
            or "patrimonio_liquido" not in self.data.columns
        ):
            return 0.0
        rs = self.data["lucro_liquido"].tail(anos) / self.data[
            "patrimonio_liquido"
        ].tail(anos)
        if len(rs) < 2:
            return 0.0
        return ((rs.iloc[-1] / rs.iloc[0]) ** (1 / (len(rs) - 1)) - 1) * 100

    def calculate_ebitda_atividade(self) -> float:
        """
        Calcula EBITDA / Receita Líquida (ou margem EBITDA).
        """
        if (
            "ebitda" not in self.data.columns
            or "receita_liquida" not in self.data.columns
        ):
            return 0.0
        e = self.data["ebitda"].iloc[-1]
        r = self.data["receita_liquida"].iloc[-1]
        if r == 0:
            return 0.0
        return (e / r) * 100

    def calculate_divida_liquida_ebitda(self) -> float:
        """
        Calcula Dívida Líquida / EBITDA
        """
        if (
            "divida_liquida" not in self.data.columns
            or "ebitda" not in self.data.columns
        ):
            return 0.0
        d = self.data["divida_liquida"].iloc[-1]
        e = self.data["ebitda"].iloc[-1]
        if e == 0:
            return 0.0
        return d / e

    def calculate_qualidade_lucros(self) -> float:
        """
        Proporção Fluxo de Caixa Operacional / Lucro Líquido.
        """
        if (
            "fluxo_caixa_operacional" not in self.data.columns
            or "lucro_liquido" not in self.data.columns
        ):
            return 0.0
        fco = self.data["fluxo_caixa_operacional"].iloc[-1]
        ll = self.data["lucro_liquido"].iloc[-1]
        if ll == 0:
            return 0.0
        return fco / ll
