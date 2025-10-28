"""
Exemplo de Uso da Biblioteca Singular Finance

Este arquivo demonstra como usar as principais funcionalidades da biblioteca
para análise financeira corporativa.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import singular_finance as sf
from singular_finance.data import get_stock_data, get_financial_statements
from singular_finance.indicators import FinancialIndicators
from singular_finance.valuation import ValuationModels
from singular_finance.cash_flow import CashFlowAnalysis
from singular_finance.visualization import FinancialCharts
from singular_finance.models import MathematicalModels
from singular_finance.utils import FinancialUtils

# Configuração para exibir gráficos
plt.rcParams['figure.figsize'] = (12, 8)
plt.style.use('seaborn-v0_8')


def exemplo_analise_completa():
    """
    Exemplo completo de análise financeira usando Singular Finance.
    """
    print("🚀 Exemplo de Análise Financeira Completa com Singular Finance")
    print("=" * 60)
    
    # 1. DADOS DE EXEMPLO
    print("\n📊 1. Criando dados de exemplo...")
    
    # Criar dados financeiros simulados para uma empresa fictícia
    np.random.seed(42)
    anos = 5
    dados_base = {
        'ano': range(2020, 2020 + anos),
        'receita_liquida': [1000000, 1100000, 1200000, 1300000, 1400000],
        'ebitda': [200000, 220000, 240000, 260000, 280000],
        'lucro_liquido': [100000, 120000, 140000, 160000, 180000],
        'ativo_total': [2000000, 2200000, 2400000, 2600000, 2800000],
        'patrimonio_liquido': [800000, 900000, 1000000, 1100000, 1200000],
        'ativo_circulante': [500000, 550000, 600000, 650000, 700000],
        'passivo_circulante': [200000, 220000, 240000, 260000, 280000],
        'passivo_total': [1200000, 1300000, 1400000, 1500000, 1600000],
        'estoque': [100000, 110000, 120000, 130000, 140000],
        'acoes_circulantes': [1000000, 1000000, 1000000, 1000000, 1000000],
        'fluxo_caixa_operacional': [150000, 160000, 170000, 180000, 190000],
        'fluxo_caixa_investimento': [-50000, -60000, -70000, -80000, -90000],
        'fluxo_caixa_financiamento': [-30000, -20000, -10000, 0, 10000]
    }
    
    df = pd.DataFrame(dados_base)
    df.set_index('ano', inplace=True)
    
    print(f"✅ Dados criados: {len(df)} anos de dados financeiros")
    
    # 2. ANÁLISE DE INDICADORES FINANCEIROS
    print("\n📈 2. Calculando indicadores financeiros...")
    
    calculator = FinancialIndicators(df)
    indicadores = calculator.calculate_all_indicators(preco_acao=10.0)
    
    print("📊 Indicadores Calculados:")
    for nome, valor in indicadores.items():
        if valor is not None:
            if 'ratio' in nome.lower() or 'liquidez' in nome.lower():
                print(f"   {nome}: {valor:.2f}")
            else:
                print(f"   {nome}: {valor:.2f}%")
    
    # 3. ANÁLISE DE VALUATION
    print("\n💰 3. Realizando análise de valuation...")
    
    valuation = ValuationModels(df)
    
    # Análise DCF
    dcf_result = valuation.dcf_analysis(
        taxa_desconto=0.10,
        taxa_crescimento_perpetuo=0.02,
        anos_projecao=5
    )
    
    print(f"📊 Valor da Empresa (DCF): R$ {dcf_result['valor_empresa']:,.0f}")
    print(f"📊 Valor Terminal: R$ {dcf_result['valor_terminal']:,.0f}")
    
    # Análise por múltiplos
    empresas_comparaveis = [
        {'ev_revenue': 2.0, 'ev_ebitda': 8.0, 'pe_ratio': 15.0, 'pb_ratio': 2.0},
        {'ev_revenue': 2.5, 'ev_ebitda': 10.0, 'pe_ratio': 18.0, 'pb_ratio': 2.5},
        {'ev_revenue': 1.8, 'ev_ebitda': 7.5, 'pe_ratio': 12.0, 'pb_ratio': 1.8}
    ]
    
    multiples_result = valuation.multiples_analysis(empresas_comparaveis, preco_acao_atual=10.0)
    
    print(f"📊 Valor por Múltiplos: R$ {multiples_result['valor_medio']:,.0f}")
    if 'preco_justo' in multiples_result:
        print(f"📊 Preço Justo da Ação: R$ {multiples_result['preco_justo']:.2f}")
        print(f"📊 Upside/Downside: {multiples_result['upside_downside']:.1f}%")
    
    # 4. ANÁLISE DE FLUXO DE CAIXA
    print("\n💸 4. Analisando fluxo de caixa...")
    
    cash_flow = CashFlowAnalysis(df)
    fcf_result = cash_flow.calculate_free_cash_flow()
    
    print(f"📊 FCF Atual: R$ {fcf_result['valor_atual']:,.0f}")
    print(f"📊 FCF Médio: R$ {fcf_result['media']:,.0f}")
    print(f"📊 Qualidade do FCF: {fcf_result['qualidade']['qualidade']}")
    
    # Análise completa de fluxo de caixa
    complete_cf = cash_flow.analyze_cash_flow_statement()
    print(f"📊 Qualidade Geral: {complete_cf['qualidade_geral']['qualidade']}")
    
    # 5. VISUALIZAÇÃO DE DADOS
    print("\n📊 5. Criando visualizações...")
    
    charts = FinancialCharts()
    
    # Gráfico de indicadores
    fig1 = charts.plot_financial_metrics(indicadores, title="Indicadores Financeiros da Empresa")
    plt.show()
    
    # Gráfico de crescimento da receita
    fig2 = charts.plot_revenue_growth(df['receita_liquida'], title="Crescimento da Receita")
    plt.show()
    
    # Dashboard interativo
    dashboard = charts.plot_interactive_dashboard(df, title="Dashboard Financeiro")
    dashboard.show()
    
    # 6. MODELOS MATEMÁTICOS
    print("\n🧮 6. Aplicando modelos matemáticos...")
    
    models = MathematicalModels()
    
    # Simulação Monte Carlo para preço da ação
    mc_result = models.monte_carlo_simulation(
        initial_price=10.0,
        drift=0.05,
        volatility=0.2,
        time_horizon=1.0,
        num_simulations=10000
    )
    
    print(f"📊 Preço Esperado (Monte Carlo): R$ {mc_result['mean_final_price']:.2f}")
    print(f"📊 VaR 95%: R$ {mc_result['var_95']:.2f}")
    
    # Cálculo de VaR
    returns = df['receita_liquida'].pct_change().dropna()
    var_result = models.var_calculation(returns, confidence_level=0.05)
    
    print(f"📊 VaR Histórico: {var_result['var']:.2%}")
    print(f"📊 CVaR: {var_result['cvar']:.2%}")
    
    # 7. UTILITÁRIOS FINANCEIROS
    print("\n🔧 7. Usando utilitários financeiros...")
    
    utils = FinancialUtils()
    
    # Formatação de valores
    valor_formatado = utils.format_currency(1400000, "BRL")
    percentual_formatado = utils.format_percentage(0.15)
    
    print(f"📊 Receita Formatada: {valor_formatado}")
    print(f"📊 Margem Formatada: {percentual_formatado}")
    
    # Cálculo de CAGR
    cagr = utils.calculate_cagr(1000000, 1400000, 4)
    print(f"📊 CAGR (4 anos): {cagr:.2%}")
    
    # 8. RELATÓRIO FINAL
    print("\n📋 8. Relatório Final:")
    print("=" * 40)
    
    print(f"🏢 Empresa: Empresa Fictícia S.A.")
    print(f"📅 Período: 2020-2024")
    print(f"💰 Valor da Empresa (DCF): R$ {dcf_result['valor_empresa']:,.0f}")
    print(f"📈 ROE: {indicadores['roe']:.1f}%")
    print(f"📊 Margem Líquida: {indicadores['margem_liquida']:.1f}%")
    print(f"💧 Liquidez Corrente: {indicadores['liquidez_corrente']:.1f}")
    print(f"💸 FCF: R$ {fcf_result['valor_atual']:,.0f}")
    print(f"🎯 Qualidade FCF: {fcf_result['qualidade']['qualidade']}")
    
    print("\n✅ Análise completa finalizada!")


def exemplo_coleta_dados():
    """
    Exemplo de coleta de dados reais do Yahoo Finance.
    """
    print("\n🌐 Exemplo de Coleta de Dados Reais")
    print("=" * 40)
    
    try:
        # Coletar dados de uma ação brasileira
        symbol = "PETR4.SA"  # Petrobras
        print(f"📊 Coletando dados para {symbol}...")
        
        # Dados históricos
        stock_data = get_stock_data(symbol, period="2y")
        print(f"✅ Dados históricos coletados: {len(stock_data)} registros")
        
        # Demonstrações financeiras
        income_statement = get_financial_statements(symbol, "income")
        print(f"✅ Demonstração de resultados coletada: {len(income_statement)} linhas")
        
        # Análise básica
        if not stock_data.empty:
            preco_atual = stock_data['Fechamento'].iloc[-1]
            preco_anterior = stock_data['Fechamento'].iloc[-2]
            variacao = (preco_atual - preco_anterior) / preco_anterior
            
            print(f"📈 Preço Atual: R$ {preco_atual:.2f}")
            print(f"📊 Variação: {variacao:.2%}")
        
    except Exception as e:
        print(f"❌ Erro na coleta de dados: {str(e)}")
        print("💡 Usando dados simulados para demonstração...")
        
        # Dados simulados como fallback
        exemplo_analise_completa()


def exemplo_analise_setorial():
    """
    Exemplo de análise comparativa entre empresas do mesmo setor.
    """
    print("\n🏭 Exemplo de Análise Setorial")
    print("=" * 40)
    
    # Dados simulados para 3 empresas do setor de tecnologia
    empresas = {
        'TechCorp': {
            'receita_liquida': [500000, 600000, 750000, 900000, 1100000],
            'lucro_liquido': [50000, 80000, 120000, 180000, 250000],
            'ativo_total': [1000000, 1200000, 1500000, 1800000, 2200000],
            'patrimonio_liquido': [400000, 500000, 650000, 800000, 1000000]
        },
        'DataSoft': {
            'receita_liquida': [300000, 400000, 550000, 700000, 850000],
            'lucro_liquido': [30000, 50000, 80000, 120000, 160000],
            'ativo_total': [800000, 1000000, 1300000, 1600000, 1900000],
            'patrimonio_liquido': [300000, 400000, 550000, 700000, 850000]
        },
        'CloudTech': {
            'receita_liquida': [200000, 300000, 450000, 650000, 900000],
            'lucro_liquido': [20000, 40000, 70000, 110000, 170000],
            'ativo_total': [600000, 800000, 1100000, 1400000, 1800000],
            'patrimonio_liquido': [200000, 300000, 450000, 600000, 800000]
        }
    }
    
    resultados_setoriais = {}
    
    for empresa, dados in empresas.items():
        df = pd.DataFrame(dados)
        df.index = range(2020, 2025)
        
        calculator = FinancialIndicators(df)
        indicadores = calculator.calculate_all_indicators()
        
        resultados_setoriais[empresa] = indicadores
        
        print(f"\n📊 {empresa}:")
        print(f"   ROE: {indicadores['roe']:.1f}%")
        print(f"   ROA: {indicadores['roa']:.1f}%")
        print(f"   Margem Líquida: {indicadores['margem_liquida']:.1f}%")
    
    # Análise comparativa
    print(f"\n📈 Análise Comparativa:")
    roe_values = [resultados_setoriais[emp]['roe'] for emp in empresas.keys()]
    melhor_roe = list(empresas.keys())[roe_values.index(max(roe_values))]
    print(f"🏆 Melhor ROE: {melhor_roe} ({max(roe_values):.1f}%)")


if __name__ == "__main__":
    print("🎯 Singular Finance - Exemplos de Uso")
    print("=" * 50)
    
    # Executar exemplos
    exemplo_analise_completa()
    exemplo_coleta_dados()
    exemplo_analise_setorial()
    
    print("\n🎉 Todos os exemplos executados com sucesso!")
    print("📚 Para mais informações, consulte a documentação da biblioteca.")
