"""
Módulo de Valuation

Este módulo contém classes e funções para análise de valuation empresarial,
incluindo modelos DCF, múltiplos comparáveis e análise de valor intrínseco.
"""

__all__ = ["ValuationModels", "dcf_analysis", "multiples_analysis", "asset_based_valuation"]

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import warnings
from scipy.optimize import minimize


class ValuationModels:
    """
    Classe principal para modelos de valuation empresarial.
    
    Esta classe fornece métodos para diferentes abordagens de valuation,
    incluindo DCF, múltiplos comparáveis e análise de valor intrínseco.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Inicializa a classe com dados financeiros.
        
        Args:
            data: DataFrame com dados financeiros históricos da empresa
        """
        self.data = data.copy()
        self._validate_data()
    
    def _validate_data(self) -> None:
        """Valida se os dados contêm as colunas necessárias."""
        required_columns = [
            'receita_liquida', 'ebitda', 'lucro_liquido', 
            'ativo_total', 'patrimonio_liquido'
        ]
        
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            warnings.warn(f"Colunas ausentes para valuation: {missing_columns}")
    
    def dcf_analysis(
        self,
        taxa_desconto: float,
        taxa_crescimento_perpetuo: float = 0.02,
        anos_projecao: int = 5,
        margem_ebitda: Optional[float] = None,
        capex_percentual: float = 0.05,
        variacao_nwc_percentual: float = 0.02,
        taxa_crescimento_receita: Optional[float] = None,
        decrescimento_crescimento: float = 0.8,
    ) -> Dict[str, Union[float, pd.DataFrame]]:
        """
        Realiza análise DCF (Discounted Cash Flow).
        
        Args:
            taxa_desconto: Taxa de desconto (WACC)
            taxa_crescimento_perpetuo: Taxa de crescimento perpetuo
            anos_projecao: Número de anos para projeção
            margem_ebitda: Margem EBITDA assumida (se None, usa média histórica)
            capex_percentual: Percentual de CAPEX sobre receita
            variacao_nwc_percentual: Variação percentual do capital de giro
            taxa_crescimento_receita: Taxa de crescimento da receita (se None, usa média histórica)
            decrescimento_crescimento: Fator de decrescimento da taxa de crescimento da receita
            
        Returns:
            Dicionário com valor da empresa e projeções
        """
        # Dados históricos mais recentes
        receita_atual = self.data['receita_liquida'].iloc[-1]
        ebitda_atual = self.data['ebitda'].iloc[-1] if 'ebitda' in self.data.columns else None
        
        # Calcular margem EBITDA histórica se não fornecida
        if margem_ebitda is None:
            if ebitda_atual is not None:
                margem_ebitda = ebitda_atual / receita_atual
            else:
                margem_ebitda = 0.15  # Margem padrão
        
        # Calcular taxa de crescimento histórica da receita se não fornecida
        if taxa_crescimento_receita is None:
            if len(self.data) > 1:
                taxa_crescimento_receita = self._calculate_growth_rate('receita_liquida')
            else:
                taxa_crescimento_receita = 0.05  # Crescimento padrão
        
        # Projeções
        projecoes = []
        receita_projetada = receita_atual
        
        for ano in range(1, anos_projecao + 1):
            # A taxa de crescimento da receita diminui ao longo do tempo
            taxa_crescimento_ano = taxa_crescimento_receita * (decrescimento_crescimento ** (ano - 1))
            receita_projetada *= (1 + taxa_crescimento_ano)
            
            # Calcular EBITDA projetado
            ebitda_projetado = receita_projetada * margem_ebitda
            
            # Calcular CAPEX como um percentual da receita
            capex = receita_projetada * capex_percentual
            
            # Calcular variação do capital de giro como um percentual da receita
            variacao_nwc = receita_projetada * variacao_nwc_percentual
            
            # Calcular fluxo de caixa livre
            fcf = ebitda_projetado - capex - variacao_nwc
            
            projecoes.append({
                'ano': ano,
                'receita': receita_projetada,
                'ebitda': ebitda_projetado,
                'capex': capex,
                'variacao_nwc': variacao_nwc,
                'fcf': fcf,
                'fcf_descontado': fcf / ((1 + taxa_desconto) ** ano)
            })
        
        # Calcular o valor terminal usando o modelo de crescimento de Gordon
        ultimo_fcf = projecoes[-1]['fcf']
        valor_terminal = (ultimo_fcf * (1 + taxa_crescimento_perpetuo)) / (taxa_desconto - taxa_crescimento_perpetuo)
        valor_terminal_descontado = valor_terminal / ((1 + taxa_desconto) ** anos_projecao)
        
        # O valor da empresa é a soma dos fluxos de caixa descontados e do valor terminal descontado
        fcf_descontado_total = sum([p['fcf_descontado'] for p in projecoes])
        valor_empresa = fcf_descontado_total + valor_terminal_descontado
        
        # Criar DataFrame com projeções
        df_projecoes = pd.DataFrame(projecoes)
        
        return {
            'valor_empresa': valor_empresa,
            'valor_terminal': valor_terminal,
            'valor_terminal_descontado': valor_terminal_descontado,
            'fcf_descontado_total': fcf_descontado_total,
            'projecoes': df_projecoes,
            'taxa_desconto': taxa_desconto,
            'taxa_crescimento_perpetuo': taxa_crescimento_perpetuo
        }
    
    def _calculate_growth_rate(self, column: str) -> float:
        """Calcula taxa de crescimento histórica."""
        if len(self.data) < 2:
            return 0.05
        
        valores = self.data[column].dropna()
        if len(valores) < 2:
            return 0.05
        
        # Calcular CAGR
        valor_inicial = valores.iloc[0]
        valor_final = valores.iloc[-1]
        anos = len(valores) - 1
        
        if valor_inicial <= 0 or anos == 0:
            return 0.05
        
        return (valor_final / valor_inicial) ** (1 / anos) - 1
    
    def comparables_analysis(
        self,
        empresas_comparaveis: List[Dict[str, Union[str, float]]],
        preco_acao_atual: Optional[float] = None
    ) -> Dict[str, Union[float, Dict]]:
        """
        Análise por múltiplos comparáveis.
        
        Args:
            empresas_comparaveis: Lista de empresas comparáveis com seus múltiplos
            preco_acao_atual: Preço atual da ação (opcional)
            
        Returns:
            Dicionário com análise de múltiplos
        """
        if not empresas_comparaveis:
            raise ValueError("Lista de empresas comparáveis não pode estar vazia")
        
        # Calcular múltiplos da empresa atual
        receita_atual = self.data['receita_liquida'].iloc[-1]
        ebitda_atual = self.data['ebitda'].iloc[-1] if 'ebitda' in self.data.columns else None
        lucro_liquido = self.data['lucro_liquido'].iloc[-1]
        patrimonio_liquido = self.data['patrimonio_liquido'].iloc[-1]
        
        # Múltiplos médios das empresas comparáveis
        ev_revenue_multiples = []
        ev_ebitda_multiples = []
        pe_multiples = []
        pb_multiples = []
        
        for empresa in empresas_comparaveis:
            if 'ev_revenue' in empresa:
                ev_revenue_multiples.append(empresa['ev_revenue'])
            if 'ev_ebitda' in empresa:
                ev_ebitda_multiples.append(empresa['ev_ebitda'])
            if 'pe_ratio' in empresa:
                pe_multiples.append(empresa['pe_ratio'])
            if 'pb_ratio' in empresa:
                pb_multiples.append(empresa['pb_ratio'])
        
        # Calcular médias
        resultados = {}
        
        if ev_revenue_multiples:
            ev_revenue_medio = np.mean(ev_revenue_multiples)
            valor_por_revenue = receita_atual * ev_revenue_medio
            resultados['valor_por_revenue'] = valor_por_revenue
            resultados['ev_revenue_multiplo'] = ev_revenue_medio
        
        if ev_ebitda_multiples and ebitda_atual is not None:
            ev_ebitda_medio = np.mean(ev_ebitda_multiples)
            valor_por_ebitda = ebitda_atual * ev_ebitda_medio
            resultados['valor_por_ebitda'] = valor_por_ebitda
            resultados['ev_ebitda_multiplo'] = ev_ebitda_medio
        
        if pe_multiples:
            pe_medio = np.mean(pe_multiples)
            valor_por_pe = lucro_liquido * pe_medio
            resultados['valor_por_pe'] = valor_por_pe
            resultados['pe_multiplo'] = pe_medio
        
        if pb_multiples:
            pb_medio = np.mean(pb_multiples)
            valor_por_pb = patrimonio_liquido * pb_medio
            resultados['valor_por_pb'] = valor_por_pb
            resultados['pb_multiplo'] = pb_medio
        
        # Calcular valor médio ponderado
        valores = [v for k, v in resultados.items() if k.startswith('valor_por_')]
        if valores:
            resultados['valor_medio'] = np.mean(valores)
        
        # Análise de preço da ação se fornecido
        if preco_acao_atual is not None and 'acoes_circulantes' in self.data.columns:
            acoes_circulantes = self.data['acoes_circulantes'].iloc[-1]
            if acoes_circulantes > 0:
                resultados['preco_acao_atual'] = preco_acao_atual
                resultados['valor_empresa_atual'] = preco_acao_atual * acoes_circulantes
                
                if 'valor_medio' in resultados:
                    preco_justo = resultados['valor_medio'] / acoes_circulantes
                    resultados['preco_justo'] = preco_justo
                    resultados['upside_downside'] = (preco_justo / preco_acao_atual - 1) * 100
        
        return resultados
    
    def asset_based_valuation(self) -> Dict[str, float]:
        """
        Valuation baseado em ativos.
        
        Returns:
            Dicionário com valores baseados em ativos
        """
        ativo_total = self.data['ativo_total'].iloc[-1]
        passivo_total = self.data['passivo_total'].iloc[-1] if 'passivo_total' in self.data.columns else 0
        patrimonio_liquido = self.data['patrimonio_liquido'].iloc[-1]
        
        # Valor contábil
        valor_contabil = patrimonio_liquido
        
        # Valor de liquidação (assumindo desconto de 20% nos ativos)
        valor_liquidacao = ativo_total * 0.8 - passivo_total
        
        # Valor de reposição (assumindo inflação de 5% ao ano)
        inflacao_anual = 0.05
        anos_historico = len(self.data)
        valor_reposicao = ativo_total * ((1 + inflacao_anual) ** anos_historico)
        
        return {
            'valor_contabil': valor_contabil,
            'valor_liquidacao': valor_liquidacao,
            'valor_reposicao': valor_reposicao,
            'ativo_total': ativo_total,
            'passivo_total': passivo_total,
            'patrimonio_liquido': patrimonio_liquido
        }
    
    def sensitivity_analysis(
        self,
        dcf_parameters: Dict[str, float],
        sensitivity_variables: Dict[str, Tuple[float, float]],
    ) -> pd.DataFrame:
        """
        Análise de sensibilidade do modelo DCF.

        Args:
            dcf_parameters: Dicionário com os parâmetros base para a análise DCF.
            sensitivity_variables: Dicionário com as variáveis para a análise de sensibilidade e seus intervalos.

        Returns:
            DataFrame com a análise de sensibilidade.
        """
        resultados = []

        # Gerar todas as combinações de parâmetros
        from itertools import product

        param_names = list(sensitivity_variables.keys())
        param_ranges = [np.linspace(start, end, 5) for start, end in sensitivity_variables.values()]

        for param_combination in product(*param_ranges):
            params = dcf_parameters.copy()
            for i, param_name in enumerate(param_names):
                params[param_name] = param_combination[i]

            try:
                dcf_result = self.dcf_analysis(**params)
                result_row = {param_name: params[param_name] for param_name in param_names}
                result_row["valor_empresa"] = dcf_result["valor_empresa"]
                resultados.append(result_row)
            except:
                continue

        return pd.DataFrame(resultados)


def dcf_analysis(
    data: pd.DataFrame,
    taxa_desconto: float,
    taxa_crescimento_perpetuo: float = 0.02,
    anos_projecao: int = 5
) -> Dict[str, Union[float, pd.DataFrame]]:
    """
    Função utilitária para análise DCF.
    
    Args:
        data: DataFrame com dados financeiros
        taxa_desconto: Taxa de desconto (WACC)
        taxa_crescimento_perpetuo: Taxa de crescimento perpetuo
        anos_projecao: Número de anos para projeção
        
    Returns:
        Dicionário com resultado da análise DCF
    """
    calculator = ValuationModels(data)
    return calculator.dcf_analysis(taxa_desconto, taxa_crescimento_perpetuo, anos_projecao)


def comparables_analysis(
    data: pd.DataFrame,
    empresas_comparaveis: List[Dict[str, Union[str, float]]],
    preco_acao_atual: Optional[float] = None
) -> Dict[str, Union[float, Dict]]:
    """
    Função utilitária para análise por múltiplos.
    
    Args:
        data: DataFrame com dados financeiros
        empresas_comparaveis: Lista de empresas comparáveis
        preco_acao_atual: Preço atual da ação
        
    Returns:
        Dicionário com resultado da análise por múltiplos
    """
    calculator = ValuationModels(data)
    return calculator.comparables_analysis(empresas_comparaveis, preco_acao_atual)


def asset_based_valuation(data: pd.DataFrame) -> Dict[str, float]:
    """
    Função utilitária para valuation baseado em ativos.
    
    Args:
        data: DataFrame com dados financeiros
        
    Returns:
        Dicionário com valores baseados em ativos
    """
    calculator = ValuationModels(data)
    return calculator.asset_based_valuation()
