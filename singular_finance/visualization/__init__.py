"""
Módulo de Visualização de Dados Financeiros

Este módulo contém classes e funções para criação de gráficos e dashboards
interativos para análise financeira.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Union, Tuple
import warnings

# Configuração do estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class FinancialCharts:
    """
    Classe principal para criação de gráficos financeiros.
    
    Esta classe fornece métodos para criação de diversos tipos de gráficos
    específicos para análise financeira.
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Inicializa a classe com dados financeiros.
        
        Args:
            data: DataFrame com dados financeiros (opcional)
        """
        self.data = data.copy() if data is not None else None
        self.figures = {}  # Armazenar figuras criadas
    
    def plot_financial_metrics(
        self,
        indicators: Dict[str, float],
        title: str = "Indicadores Financeiros",
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Cria gráfico de barras com indicadores financeiros.
        
        Args:
            indicators: Dicionário com indicadores financeiros
            title: Título do gráfico
            figsize: Tamanho da figura
            
        Returns:
            Figura matplotlib
        """
        # Filtrar valores válidos
        valid_indicators = {k: v for k, v in indicators.items() if v is not None and not np.isnan(v)}
        
        if not valid_indicators:
            raise ValueError("Nenhum indicador válido fornecido")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Preparar dados
        labels = list(valid_indicators.keys())
        values = list(valid_indicators.values())
        
        # Criar gráfico de barras
        bars = ax.bar(labels, values, color=plt.cm.viridis(np.linspace(0, 1, len(labels))))
        
        # Personalizar gráfico
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylabel('Valor', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Armazenar figura
        self.figures['financial_metrics'] = fig
        
        return fig
    
    def plot_revenue_growth(
        self,
        revenue_data: pd.Series,
        title: str = "Crescimento da Receita",
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Cria gráfico de crescimento da receita ao longo do tempo.
        
        Args:
            revenue_data: Série temporal com dados de receita
            title: Título do gráfico
            figsize: Tamanho da figura
            
        Returns:
            Figura matplotlib
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Gráfico da receita absoluta
        ax1.plot(revenue_data.index, revenue_data.values, marker='o', linewidth=2, markersize=6)
        ax1.set_title(f'{title} - Valores Absolutos', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Receita (R$)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico do crescimento percentual
        growth_rates = revenue_data.pct_change() * 100
        ax2.bar(growth_rates.index[1:], growth_rates.values[1:], 
                color='green', alpha=0.7, width=0.8)
        ax2.set_title('Taxa de Crescimento Anual (%)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Crescimento (%)', fontsize=12)
        ax2.set_xlabel('Período', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Adicionar linha de crescimento médio
        avg_growth = growth_rates.mean()
        ax2.axhline(y=avg_growth, color='red', linestyle='--', 
                   label=f'Crescimento Médio: {avg_growth:.1f}%')
        ax2.legend()
        
        plt.tight_layout()
        
        # Armazenar figura
        self.figures['revenue_growth'] = fig
        
        return fig
    
    def plot_cash_flow_waterfall(
        self,
        cash_flow_data: Dict[str, float],
        title: str = "Análise de Fluxo de Caixa",
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Cria gráfico waterfall do fluxo de caixa.
        
        Args:
            cash_flow_data: Dicionário com componentes do fluxo de caixa
            title: Título do gráfico
            figsize: Tamanho da figura
            
        Returns:
            Figura matplotlib
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Preparar dados para waterfall
        components = list(cash_flow_data.keys())
        values = list(cash_flow_data.values())
        
        # Calcular posições cumulativas
        cumulative = np.cumsum([0] + values)
        
        # Criar gráfico de barras
        colors = ['green' if v >= 0 else 'red' for v in values]
        bars = ax.bar(range(len(components)), values, color=colors, alpha=0.7)
        
        # Adicionar linhas de conexão
        for i in range(len(cumulative) - 1):
            ax.plot([i, i+1], [cumulative[i], cumulative[i+1]], 'k-', linewidth=2)
        
        # Personalizar gráfico
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylabel('Valor (R$)', fontsize=12)
        ax.set_xticks(range(len(components)))
        ax.set_xticklabels(components, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:,.0f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        
        # Armazenar figura
        self.figures['cash_flow_waterfall'] = fig
        
        return fig
    
    def plot_valuation_sensitivity(
        self,
        sensitivity_data: pd.DataFrame,
        title: str = "Análise de Sensibilidade - DCF",
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Cria heatmap de análise de sensibilidade.
        
        Args:
            sensitivity_data: DataFrame com dados de sensibilidade
            title: Título do gráfico
            figsize: Tamanho da figura
            
        Returns:
            Figura matplotlib
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Criar pivot table para heatmap
        pivot_data = sensitivity_data.pivot(
            index='taxa_crescimento', 
            columns='taxa_desconto', 
            values='valor_empresa'
        )
        
        # Criar heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='RdYlGn', 
                   ax=ax, cbar_kws={'label': 'Valor da Empresa (R$)'})
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Taxa de Desconto (%)', fontsize=12)
        ax.set_ylabel('Taxa de Crescimento (%)', fontsize=12)
        
        plt.tight_layout()
        
        # Armazenar figura
        self.figures['valuation_sensitivity'] = fig
        
        return fig
    
    def plot_interactive_dashboard(
        self,
        data: pd.DataFrame,
        title: str = "Dashboard Financeiro Interativo"
    ) -> go.Figure:
        """
        Cria dashboard interativo com Plotly.
        
        Args:
            data: DataFrame com dados financeiros
            title: Título do dashboard
            
        Returns:
            Figura Plotly
        """
        # Criar subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Receita vs EBITDA', 'Fluxo de Caixa', 
                          'Indicadores de Rentabilidade', 'Análise de Liquidez'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Gráfico 1: Receita vs EBITDA
        if 'receita_liquida' in data.columns and 'ebitda' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['receita_liquida'], 
                          name='Receita Líquida', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['ebitda'], 
                          name='EBITDA', line=dict(color='red')),
                row=1, col=1, secondary_y=True
            )
        
        # Gráfico 2: Fluxo de Caixa
        if 'fluxo_caixa_operacional' in data.columns:
            fig.add_trace(
                go.Bar(x=data.index, y=data['fluxo_caixa_operacional'], 
                      name='FCO', marker_color='green'),
                row=1, col=2
            )
        
        # Gráfico 3: ROE e ROA
        if 'lucro_liquido' in data.columns and 'patrimonio_liquido' in data.columns:
            roe = (data['lucro_liquido'] / data['patrimonio_liquido'] * 100).fillna(0)
            fig.add_trace(
                go.Scatter(x=data.index, y=roe, name='ROE (%)', 
                          line=dict(color='purple')),
                row=2, col=1
            )
        
        # Gráfico 4: Liquidez Corrente
        if 'ativo_circulante' in data.columns and 'passivo_circulante' in data.columns:
            liquidez = (data['ativo_circulante'] / data['passivo_circulante']).fillna(0)
            fig.add_trace(
                go.Scatter(x=data.index, y=liquidez, name='Liquidez Corrente', 
                          line=dict(color='orange')),
                row=2, col=2
            )
        
        # Atualizar layout
        fig.update_layout(
            title_text=title,
            showlegend=True,
            height=800,
            template='plotly_white'
        )
        
        # Armazenar figura
        self.figures['interactive_dashboard'] = fig
        
        return fig
    
    def plot_correlation_matrix(
        self,
        data: pd.DataFrame,
        title: str = "Matriz de Correlação Financeira",
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Cria matriz de correlação dos indicadores financeiros.
        
        Args:
            data: DataFrame com dados financeiros
            title: Título do gráfico
            figsize: Tamanho da figura
            
        Returns:
            Figura matplotlib
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calcular matriz de correlação
        correlation_matrix = data.corr()
        
        # Criar heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax, cbar_kws={'label': 'Correlação'})
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Armazenar figura
        self.figures['correlation_matrix'] = fig
        
        return fig
    
    def plot_financial_ratios_trend(
        self,
        ratios_data: Dict[str, pd.Series],
        title: str = "Evolução dos Indicadores Financeiros",
        figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        Cria gráfico de evolução dos indicadores financeiros.
        
        Args:
            ratios_data: Dicionário com séries temporais dos indicadores
            title: Título do gráfico
            figsize: Tamanho da figura
            
        Returns:
            Figura matplotlib
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Plotar cada indicador
        for i, (name, series) in enumerate(ratios_data.items()):
            if i < len(axes):
                ax = axes[i]
                ax.plot(series.index, series.values, marker='o', linewidth=2)
                ax.set_title(name, fontsize=12, fontweight='bold')
                ax.set_ylabel('Valor', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
        
        # Ocultar subplots não utilizados
        for i in range(len(ratios_data), len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Armazenar figura
        self.figures['ratios_trend'] = fig
        
        return fig
    
    def save_figures(self, directory: str = "charts"):
        """
        Salva todas as figuras criadas.
        
        Args:
            directory: Diretório para salvar as figuras
        """
        import os
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for name, fig in self.figures.items():
            if hasattr(fig, 'savefig'):  # Matplotlib figure
                fig.savefig(f"{directory}/{name}.png", dpi=300, bbox_inches='tight')
            else:  # Plotly figure
                fig.write_html(f"{directory}/{name}.html")
    
    def show_all_figures(self):
        """Exibe todas as figuras criadas."""
        for name, fig in self.figures.items():
            if hasattr(fig, 'show'):  # Matplotlib figure
                fig.show()
            else:  # Plotly figure
                fig.show()


def plot_financial_metrics(indicators: Dict[str, float], **kwargs) -> plt.Figure:
    """
    Função utilitária para criar gráfico de indicadores financeiros.
    
    Args:
        indicators: Dicionário com indicadores financeiros
        **kwargs: Argumentos adicionais para o gráfico
        
    Returns:
        Figura matplotlib
    """
    chart = FinancialCharts()
    return chart.plot_financial_metrics(indicators, **kwargs)


def plot_revenue_growth(revenue_data: pd.Series, **kwargs) -> plt.Figure:
    """
    Função utilitária para criar gráfico de crescimento da receita.
    
    Args:
        revenue_data: Série temporal com dados de receita
        **kwargs: Argumentos adicionais para o gráfico
        
    Returns:
        Figura matplotlib
    """
    chart = FinancialCharts()
    return chart.plot_revenue_growth(revenue_data, **kwargs)


def plot_cash_flow_waterfall(cash_flow_data: Dict[str, float], **kwargs) -> plt.Figure:
    """
    Função utilitária para criar gráfico waterfall do fluxo de caixa.
    
    Args:
        cash_flow_data: Dicionário com componentes do fluxo de caixa
        **kwargs: Argumentos adicionais para o gráfico
        
    Returns:
        Figura matplotlib
    """
    chart = FinancialCharts()
    return chart.plot_cash_flow_waterfall(cash_flow_data, **kwargs)


def create_interactive_dashboard(data: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Função utilitária para criar dashboard interativo.
    
    Args:
        data: DataFrame com dados financeiros
        **kwargs: Argumentos adicionais para o dashboard
        
    Returns:
        Figura Plotly
    """
    chart = FinancialCharts()
    return chart.plot_interactive_dashboard(data, **kwargs)
