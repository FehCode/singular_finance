"""
Módulo de Análise de Fluxo de Caixa

Este módulo contém classes e funções para análise detalhada de fluxos de caixa,
incluindo fluxos operacionais, de investimento e de financiamento.
"""

__all__ = ["CashFlowAnalysis", "analyze_cash_flow", "calculate_free_cash_flow"]

import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Optional, Union, Tuple
import warnings


class CashFlowAnalysis:
    """
    Classe principal para análise de fluxo de caixa.
    
    Esta classe fornece métodos para análise detalhada dos fluxos de caixa
    operacionais, de investimento e de financiamento.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Inicializa a classe com dados financeiros.
        
        Args:
            data: DataFrame com dados de fluxo de caixa
        """
        self.data = data.copy()
        self._validate_data()
    
    def _validate_data(self) -> None:
        """Valida se os dados contêm as colunas necessárias."""
        required_columns = [
            'fluxo_caixa_operacional', 'fluxo_caixa_investimento', 
            'fluxo_caixa_financiamento'
        ]
        
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            warnings.warn(f"Colunas ausentes para análise de fluxo de caixa: {missing_columns}")
    
    def _analyze_cash_flow_component(
        self, component_name: str, analysis_functions: Dict[str, Callable]
    ) -> Dict[str, Union[float, pd.Series]]:
        """
        Analisa um componente de fluxo de caixa de forma genérica.

        Args:
            component_name: Nome da coluna do componente de fluxo de caixa
            analysis_functions: Dicionário de funções de análise específicas do componente

        Returns:
            Dicionário com a análise do componente
        """
        if component_name not in self.data.columns:
            raise ValueError(f"Coluna '{component_name}' é necessária")

        series = self.data[component_name]

        # Estatísticas básicas
        mean_value = series.mean()
        std_value = series.std()
        growth_rate = self._calculate_growth_rate(series)

        analysis = {
            "valor_atual": series.iloc[-1],
            "media": mean_value,
            "desvio_padrao": std_value,
            "crescimento_anual": growth_rate,
            "serie_historica": series,
        }

        # Análises específicas do componente
        for name, func in analysis_functions.items():
            analysis[name] = func(series)

        return analysis

    def analyze_operational_cash_flow(self) -> Dict[str, Union[float, pd.Series]]:
        """
        Analisa o fluxo de caixa operacional.
        
        Returns:
            Dicionário com análise do fluxo operacional
        """
        analysis_functions = {
            "tendencia_positiva": lambda s: (s.iloc[-1] > s.iloc[0]) if len(s) > 1 else False,
            "volatilidade": lambda s: s.std() / abs(s.mean()) if s.mean() != 0 else 0,
            "sazonalidade": self._analyze_seasonality,
        }
        return self._analyze_cash_flow_component(
            "fluxo_caixa_operacional", analysis_functions
        )

    def analyze_investment_cash_flow(self) -> Dict[str, Union[float, pd.Series]]:
        """
        Analisa o fluxo de caixa de investimento.
        
        Returns:
            Dicionário com análise do fluxo de investimento
        """
        analysis_functions = {
            "intensidade_investimento": self._analyze_investment_intensity,
            "investimento_consistente": self._analyze_consistency,
        }
        return self._analyze_cash_flow_component(
            "fluxo_caixa_investimento", analysis_functions
        )

    def analyze_financing_cash_flow(self) -> Dict[str, Union[float, pd.Series]]:
        """
        Analisa o fluxo de caixa de financiamento.
        
        Returns:
            Dicionário com análise do fluxo de financiamento
        """
        analysis_functions = {
            "dependencia_financiamento": self._analyze_financing_dependency,
            "padrao_financiamento": self._analyze_financing_pattern,
        }
        return self._analyze_cash_flow_component(
            "fluxo_caixa_financiamento", analysis_functions
        )

    def _analyze_investment_intensity(self, series: pd.Series) -> Optional[float]:
        """Analisa a intensidade do investimento."""
        if "receita_liquida" in self.data.columns:
            receita_media = self.data["receita_liquida"].mean()
            return abs(series.mean()) / receita_media if receita_media != 0 else 0
        return None

    def _analyze_financing_dependency(self, series: pd.Series) -> Optional[float]:
        """Analisa a dependência de financiamento."""
        if "fluxo_caixa_operacional" in self.data.columns:
            fco_medio = self.data["fluxo_caixa_operacional"].mean()
            return abs(series.mean()) / abs(fco_medio) if fco_medio != 0 else 0
        return None

    
    def calculate_free_cash_flow(self) -> Dict[str, Union[float, pd.Series]]:
        """
        Calcula o fluxo de caixa livre (FCF).
        
        FCF = Fluxo de Caixa Operacional - CAPEX
        
        Returns:
            Dicionário com análise do FCF
        """
        if 'fluxo_caixa_operacional' not in self.data.columns:
            raise ValueError("Coluna 'fluxo_caixa_operacional' é necessária")
        
        fco = self.data['fluxo_caixa_operacional']
        
        # CAPEX pode ser negativo do fluxo de investimento
        if 'fluxo_caixa_investimento' in self.data.columns:
            capex = -self.data['fluxo_caixa_investimento']  # CAPEX é negativo no FCI
        else:
            capex = pd.Series([0] * len(fco), index=fco.index)
        
        # Calcular FCF
        fcf = fco - capex
        
        # Análise do FCF
        fcf_medio = fcf.mean()
        fcf_std = fcf.std()
        fcf_crescimento = self._calculate_growth_rate(fcf)
        
        # Análise de qualidade do FCF
        qualidade_fcf = self._analyze_fcf_quality(fcf, fco)
        
        return {
            'valor_atual': float(fcf.iloc[-1]) if not pd.isna(fcf.iloc[-1]) else None,
            'media': float(fcf_medio) if not pd.isna(fcf_medio) else None,
            'desvio_padrao': float(fcf_std) if not pd.isna(fcf_std) else None,
            'crescimento_anual': float(fcf_crescimento) if not pd.isna(fcf_crescimento) else None,
            'qualidade': qualidade_fcf,
            'serie_historica': fcf,
            'capex_medio': float(capex.mean()) if not pd.isna(capex.mean()) else None
        }
    
    def analyze_cash_flow_statement(self) -> Dict[str, Union[float, pd.DataFrame]]:
        """
        Análise completa da demonstração de fluxo de caixa.
        
        Returns:
            Dicionário com análise completa
        """
        resultados = {}
        
        # Análise de cada componente
        try:
            resultados['operacional'] = self.analyze_operational_cash_flow()
        except ValueError as e:
            resultados['operacional'] = {'erro': str(e)}
        
        try:
            resultados['investimento'] = self.analyze_investment_cash_flow()
        except ValueError as e:
            resultados['investimento'] = {'erro': str(e)}
        
        try:
            resultados['financiamento'] = self.analyze_financing_cash_flow()
        except ValueError as e:
            resultados['financiamento'] = {'erro': str(e)}
        
        try:
            resultados['fcf'] = self.calculate_free_cash_flow()
        except ValueError as e:
            resultados['fcf'] = {'erro': str(e)}
        
        # Análise de qualidade geral
        resultados['qualidade_geral'] = self._analyze_overall_quality()
        
        # Criar resumo executivo
        resultados['resumo'] = self._create_executive_summary(resultados)
        
        return resultados
    
    def _calculate_growth_rate(self, series: pd.Series) -> float:
        """Calcula taxa de crescimento anual com proteção contra valores inválidos."""
        if len(series) < 2:
            return 0.0

        valores = series.dropna()
        if len(valores) < 2:
            return 0.0

        valor_inicial = valores.iloc[0]
        valor_final = valores.iloc[-1]
        anos = len(valores) - 1

        # Proteções contra divisão/raízes inválidas
        try:
            if not np.isfinite(valor_inicial) or not np.isfinite(valor_final):
                return 0.0
            if valor_inicial <= 0 or anos <= 0:
                return 0.0

            # Evitar avisos de numpy ao elevar bases negativas a potências fracionárias
            if valor_inicial > 0 and valor_final > 0:
                return (valor_final / valor_inicial) ** (1 / anos) - 1
            return 0.0
        except (ZeroDivisionError, ValueError, OverflowError):
            return 0.0
    
    def _analyze_seasonality(
        self, series: pd.Series, min_data_points: int = 12, seasonality_threshold: float = 0.2
    ) -> Dict[str, float]:
        """
        Analisa a sazonalidade nos dados de uma série temporal.

        Args:
            series: A série temporal a ser analisada.
            min_data_points: O número mínimo de pontos de dados para realizar a análise.
            seasonality_threshold: O limiar para determinar se a sazonalidade é detectada.

        Returns:
            Um dicionário contendo se a sazonalidade foi detectada e a intensidade da sazonalidade.
        """
        if len(series) < min_data_points:
            return {"detectada": False, "intensidade": 0}

        # Análise simples de sazonalidade
        valores = series.dropna()
        if len(valores) < 4:
            return {"detectada": False, "intensidade": 0}

        # Calcular coeficiente de variação por trimestre
        q1 = valores[::4].std() / abs(valores[::4].mean()) if valores[::4].mean() != 0 else 0
        q2 = valores[1::4].std() / abs(valores[1::4].mean()) if valores[1::4].mean() != 0 else 0
        q3 = valores[2::4].std() / abs(valores[2::4].mean()) if valores[2::4].mean() != 0 else 0
        q4 = valores[3::4].std() / abs(valores[3::4].mean()) if valores[3::4].mean() != 0 else 0

        intensidade = max(q1, q2, q3, q4)

        return {
            "detectada": intensidade > seasonality_threshold,
            "intensidade": intensidade,
        }

    def _analyze_consistency(
        self, series: pd.Series, consistency_threshold: float = 0.5
    ) -> Dict[str, Union[bool, float]]:
        """
        Analisa a consistência de uma série temporal.

        Args:
            series: A série temporal a ser analisada.
            consistency_threshold: O limiar para determinar se a série é consistente.

        Returns:
            Um dicionário contendo se a série é consistente e a volatilidade.
        """
        valores = series.dropna()
        if len(valores) < 2:
            return {"consistente": True, "volatilidade": 0}

        # Calcular coeficiente de variação
        cv = valores.std() / abs(valores.mean()) if valores.mean() != 0 else 0

        return {
            "consistente": cv < consistency_threshold,  # Baixa volatilidade
            "volatilidade": cv,
        }
    
    def _analyze_financing_pattern(self, series: pd.Series) -> Dict[str, Union[str, float]]:
        """Analisa padrão de financiamento."""
        valores = series.dropna()
        if len(valores) == 0:
            return {'padrao': 'indefinido', 'tendencia': 0}
        
        # Contar fluxos positivos vs negativos
        positivos = (valores > 0).sum()
        negativos = (valores < 0).sum()
        neutros = (valores == 0).sum()
        
        total = len(valores)
        
        if positivos / total > 0.6:
            padrao = 'captacao_constante'
        elif negativos / total > 0.6:
            padrao = 'pagamento_constante'
        else:
            padrao = 'misto'
        
        # Calcular tendência
        if len(valores) > 1:
            tendencia = (valores.iloc[-1] - valores.iloc[0]) / abs(valores.iloc[0]) if valores.iloc[0] != 0 else 0
        else:
            tendencia = 0
        
        return {
            'padrao': padrao,
            'tendencia': tendencia,
            'percentual_positivos': positivos / total,
            'percentual_negativos': negativos / total
        }
    
    def _analyze_fcf_quality(self, fcf: pd.Series, fco: pd.Series) -> Dict[str, Union[str, float]]:
        """Analisa qualidade do FCF."""
        if len(fcf) == 0 or len(fco) == 0:
            return {'qualidade': 'indefinida', 'score': 0}
        
        # Calcular score de qualidade
        score = 0
        
        # FCF positivo consistentemente
        fcf_positivos = (fcf > 0).sum() / len(fcf)
        score += fcf_positivos * 40
        
        # FCF crescente
        if len(fcf) > 1:
            crescimento_fcf = self._calculate_growth_rate(fcf)
            score += min(crescimento_fcf * 100, 30)  # Máximo 30 pontos
        
        # FCF próximo ao FCO (baixo CAPEX)
        if len(fco) > 0:
            relacao_fcf_fco = fcf.mean() / fco.mean() if fco.mean() != 0 else 0
            score += min(relacao_fcf_fco * 30, 30)  # Máximo 30 pontos
        
        # Classificar qualidade
        if score >= 80:
            qualidade = 'excelente'
        elif score >= 60:
            qualidade = 'boa'
        elif score >= 40:
            qualidade = 'regular'
        else:
            qualidade = 'ruim'
        
        return {
            'qualidade': qualidade,
            'score': score,
            'fcf_positivos_percentual': fcf_positivos * 100
        }
    
    def _analyze_overall_quality(self) -> Dict[str, Union[str, float]]:
        """Analisa qualidade geral do fluxo de caixa."""
        scores = []
        
        # Analisar cada componente
        componentes = ['fluxo_caixa_operacional', 'fluxo_caixa_investimento', 'fluxo_caixa_financiamento']
        
        for componente in componentes:
            if componente in self.data.columns:
                serie = self.data[componente]
                if len(serie) > 0:
                    # Score baseado na consistência e crescimento
                    crescimento = self._calculate_growth_rate(serie)
                    consistencia = 1 - (serie.std() / abs(serie.mean())) if serie.mean() != 0 else 0
                    
                    score_componente = (crescimento * 50 + consistencia * 50)
                    scores.append(max(0, min(100, score_componente)))
        
        if not scores:
            return {'qualidade': 'indefinida', 'score': 0}
        
        score_medio = np.mean(scores)
        
        if score_medio >= 80:
            qualidade = 'excelente'
        elif score_medio >= 60:
            qualidade = 'boa'
        elif score_medio >= 40:
            qualidade = 'regular'
        else:
            qualidade = 'ruim'
        
        return {
            'qualidade': qualidade,
            'score': score_medio,
            'componentes_analisados': len(scores)
        }
    
    def _create_executive_summary(self, resultados: Dict) -> Dict[str, str]:
        """Cria resumo executivo da análise."""
        resumo = {}
        
        # Resumo operacional
        if 'operacional' in resultados and 'erro' not in resultados['operacional']:
            fco = resultados['operacional']
            if fco['tendencia_positiva']:
                resumo['operacional'] = f"FCO positivo com tendência crescente ({fco['crescimento_anual']*100:.1f}% a.a.)"
            else:
                resumo['operacional'] = f"FCO com tendência decrescente ({fco['crescimento_anual']*100:.1f}% a.a.)"
        
        # Resumo FCF
        if 'fcf' in resultados and 'erro' not in resultados['fcf']:
            fcf = resultados['fcf']
            resumo['fcf'] = f"FCF {fcf['qualidade']['qualidade']} com score {fcf['qualidade']['score']:.1f}"
        
        # Resumo geral
        if 'qualidade_geral' in resultados:
            qualidade = resultados['qualidade_geral']
            resumo['geral'] = f"Qualidade geral: {qualidade['qualidade']} (score: {qualidade['score']:.1f})"
        
        return resumo


def analyze_cash_flow(data: pd.DataFrame) -> Dict[str, Union[float, pd.DataFrame]]:
    """
    Função utilitária para análise completa de fluxo de caixa.
    
    Args:
        data: DataFrame com dados de fluxo de caixa
        
    Returns:
        Dicionário com análise completa
    """
    analyzer = CashFlowAnalysis(data)
    return analyzer.analyze_cash_flow_statement()


def calculate_free_cash_flow(data: pd.DataFrame) -> Dict[str, Union[float, pd.Series]]:
    """
    Função utilitária para cálculo do FCF.
    
    Args:
        data: DataFrame com dados de fluxo de caixa
        
    Returns:
        Dicionário com análise do FCF
    """
    analyzer = CashFlowAnalysis(data)
    return analyzer.calculate_free_cash_flow()
