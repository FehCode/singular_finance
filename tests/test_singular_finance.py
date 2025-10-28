"""
Testes unitários para Singular Finance

Este módulo contém testes para todas as funcionalidades da biblioteca.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Adicionar o diretório raiz ao path para importar os módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from singular_finance.indicators import FinancialIndicators
from singular_finance.valuation import ValuationModels
from singular_finance.cash_flow import CashFlowAnalysis
from singular_finance.models import MathematicalModels
from singular_finance.utils import FinancialUtils
from singular_finance.data import DataCollector, DataProcessor


class TestFinancialIndicators(unittest.TestCase):
    """Testes para o módulo de indicadores financeiros."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        self.sample_data = pd.DataFrame({
            'receita_liquida': [1000000, 1100000, 1200000],
            'lucro_liquido': [100000, 120000, 140000],
            'ativo_total': [2000000, 2200000, 2400000],
            'patrimonio_liquido': [800000, 900000, 1000000],
            'ativo_circulante': [500000, 550000, 600000],
            'passivo_circulante': [200000, 220000, 240000],
            'ebitda': [200000, 220000, 240000],
            'estoque': [100000, 110000, 120000],
            'passivo_total': [1200000, 1300000, 1400000],
            'acoes_circulantes': [1000000, 1000000, 1000000]
        })
    
    def test_roe_calculation(self):
        """Testa cálculo de ROE."""
        calculator = FinancialIndicators(self.sample_data)
        roe = calculator.calculate_roe()
        
        expected_roe = (140000 / 1000000) * 100  # 14%
        self.assertAlmostEqual(roe, expected_roe, places=2)
    
    def test_roa_calculation(self):
        """Testa cálculo de ROA."""
        calculator = FinancialIndicators(self.sample_data)
        roa = calculator.calculate_roa()
        
        expected_roa = (140000 / 2400000) * 100  # ~5.83%
        self.assertAlmostEqual(roa, expected_roa, places=2)
    
    def test_margem_liquida_calculation(self):
        """Testa cálculo de margem líquida."""
        calculator = FinancialIndicators(self.sample_data)
        margem = calculator.calculate_margem_liquida()
        
        expected_margem = (140000 / 1200000) * 100  # ~11.67%
        self.assertAlmostEqual(margem, expected_margem, places=2)
    
    def test_liquidez_corrente_calculation(self):
        """Testa cálculo de liquidez corrente."""
        calculator = FinancialIndicators(self.sample_data)
        liquidez = calculator.calculate_liquidez_corrente()
        
        expected_liquidez = 600000 / 240000  # 2.5
        self.assertAlmostEqual(liquidez, expected_liquidez, places=2)
    
    def test_all_indicators_calculation(self):
        """Testa cálculo de todos os indicadores."""
        calculator = FinancialIndicators(self.sample_data)
        indicators = calculator.calculate_all_indicators(preco_acao=10.0)
        
        self.assertIsInstance(indicators, dict)
        self.assertIn('roe', indicators)
        self.assertIn('roa', indicators)
        self.assertIn('margem_liquida', indicators)
        self.assertIn('liquidez_corrente', indicators)


class TestValuationModels(unittest.TestCase):
    """Testes para o módulo de valuation."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        self.sample_data = pd.DataFrame({
            'receita_liquida': [1000000, 1100000, 1200000, 1300000, 1400000],
            'ebitda': [200000, 220000, 240000, 260000, 280000],
            'lucro_liquido': [100000, 120000, 140000, 160000, 180000],
            'ativo_total': [2000000, 2200000, 2400000, 2600000, 2800000],
            'patrimonio_liquido': [800000, 900000, 1000000, 1100000, 1200000]
        })
    
    def test_dcf_analysis(self):
        """Testa análise DCF."""
        calculator = ValuationModels(self.sample_data)
        dcf_result = calculator.dcf_analysis(
            taxa_desconto=0.10,
            taxa_crescimento_perpetuo=0.02,
            anos_projecao=5
        )
        
        self.assertIsInstance(dcf_result, dict)
        self.assertIn('valor_empresa', dcf_result)
        self.assertIn('projecoes', dcf_result)
        self.assertGreater(dcf_result['valor_empresa'], 0)
    
    def test_multiples_analysis(self):
        """Testa análise por múltiplos."""
        calculator = ValuationModels(self.sample_data)
        
        empresas_comparaveis = [
            {'ev_revenue': 2.0, 'ev_ebitda': 8.0, 'pe_ratio': 15.0, 'pb_ratio': 2.0},
            {'ev_revenue': 2.5, 'ev_ebitda': 10.0, 'pe_ratio': 18.0, 'pb_ratio': 2.5}
        ]
        
        multiples_result = calculator.multiples_analysis(empresas_comparaveis)
        
        self.assertIsInstance(multiples_result, dict)
        self.assertIn('valor_medio', multiples_result)
        self.assertGreater(multiples_result['valor_medio'], 0)
    
    def test_asset_based_valuation(self):
        """Testa valuation baseado em ativos."""
        calculator = ValuationModels(self.sample_data)
        asset_result = calculator.asset_based_valuation()
        
        self.assertIsInstance(asset_result, dict)
        self.assertIn('valor_contabil', asset_result)
        self.assertIn('valor_liquidacao', asset_result)


class TestCashFlowAnalysis(unittest.TestCase):
    """Testes para o módulo de fluxo de caixa."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        self.sample_data = pd.DataFrame({
            'fluxo_caixa_operacional': [150000, 160000, 170000, 180000, 190000],
            'fluxo_caixa_investimento': [-50000, -60000, -70000, -80000, -90000],
            'fluxo_caixa_financiamento': [-30000, -20000, -10000, 0, 10000],
            'receita_liquida': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
    
    def test_operational_cash_flow_analysis(self):
        """Testa análise do fluxo operacional."""
        analyzer = CashFlowAnalysis(self.sample_data)
        fco_result = analyzer.analyze_operational_cash_flow()
        
        self.assertIsInstance(fco_result, dict)
        self.assertIn('valor_atual', fco_result)
        self.assertIn('media', fco_result)
        self.assertIn('crescimento_anual', fco_result)
    
    def test_free_cash_flow_calculation(self):
        """Testa cálculo do FCF."""
        analyzer = CashFlowAnalysis(self.sample_data)
        fcf_result = analyzer.calculate_free_cash_flow()
        
        self.assertIsInstance(fcf_result, dict)
        self.assertIn('valor_atual', fcf_result)
        self.assertIn('qualidade', fcf_result)
    
    def test_complete_cash_flow_analysis(self):
        """Testa análise completa de fluxo de caixa."""
        analyzer = CashFlowAnalysis(self.sample_data)
        complete_result = analyzer.analyze_cash_flow_statement()
        
        self.assertIsInstance(complete_result, dict)
        self.assertIn('operacional', complete_result)
        self.assertIn('investimento', complete_result)
        self.assertIn('financiamento', complete_result)
        self.assertIn('fcf', complete_result)


class TestMathematicalModels(unittest.TestCase):
    """Testes para o módulo de modelos matemáticos."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        self.models = MathematicalModels()
    
    def test_black_scholes_pricing(self):
        """Testa precificação Black-Scholes."""
        result = self.models.black_scholes_option_pricing(
            spot_price=100,
            strike_price=100,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option_type='call'
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('option_price', result)
        self.assertIn('delta', result)
        self.assertIn('gamma', result)
        self.assertGreater(result['option_price'], 0)
    
    def test_monte_carlo_simulation(self):
        """Testa simulação Monte Carlo."""
        result = self.models.monte_carlo_simulation(
            initial_price=100,
            drift=0.05,
            volatility=0.2,
            time_horizon=1.0,
            num_simulations=1000
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('price_paths', result)
        self.assertIn('final_prices', result)
        self.assertIn('mean_final_price', result)
        self.assertEqual(len(result['final_prices']), 1000)
    
    def test_var_calculation(self):
        """Testa cálculo de VaR."""
        returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
        result = self.models.var_calculation(returns, confidence_level=0.05)
        
        self.assertIsInstance(result, dict)
        self.assertIn('var', result)
        self.assertIn('cvar', result)
        self.assertIn('max_drawdown', result)


class TestFinancialUtils(unittest.TestCase):
    """Testes para o módulo de utilitários."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        self.utils = FinancialUtils()
    
    def test_currency_formatting(self):
        """Testa formatação de moeda."""
        formatted = self.utils.format_currency(1000000, "BRL")
        self.assertIn("R$", formatted)
        self.assertIn("1.00M", formatted)
    
    def test_percentage_formatting(self):
        """Testa formatação de percentual."""
        formatted = self.utils.format_percentage(0.15)
        self.assertIn("15.00%", formatted)
    
    def test_cagr_calculation(self):
        """Testa cálculo de CAGR."""
        cagr = self.utils.calculate_cagr(100000, 200000, 5)
        expected_cagr = (200000 / 100000) ** (1/5) - 1
        self.assertAlmostEqual(cagr, expected_cagr, places=4)
    
    def test_npv_calculation(self):
        """Testa cálculo de NPV."""
        cash_flows = [10000, 15000, 20000, 25000]
        npv = self.utils.calculate_npv(cash_flows, 0.1, 50000)
        self.assertIsInstance(npv, float)
    
    def test_sharpe_ratio_calculation(self):
        """Testa cálculo de Sharpe Ratio."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        sharpe = self.utils.calculate_sharpe_ratio(returns, 0.02)
        self.assertIsInstance(sharpe, float)


class TestDataCollector(unittest.TestCase):
    """Testes para o módulo de coleta de dados."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        self.collector = DataCollector()
    
    @patch('yfinance.Ticker')
    def test_yahoo_finance_data_collection(self, mock_ticker):
        """Testa coleta de dados do Yahoo Finance."""
        # Mock dos dados
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        result = self.collector.get_yahoo_finance_data("PETR4.SA")
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)


class TestDataProcessor(unittest.TestCase):
    """Testes para o módulo de processamento de dados."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        self.processor = DataProcessor()
        self.sample_data = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [10, 20, 30, np.inf, 50],
            'col3': [100, 200, 300, 400, 500]
        })
    
    def test_data_cleaning(self):
        """Testa limpeza de dados."""
        cleaned = self.processor.clean_financial_data(self.sample_data)
        
        self.assertIsInstance(cleaned, pd.DataFrame)
        self.assertFalse(np.isinf(cleaned).any().any())
    
    def test_returns_calculation(self):
        """Testa cálculo de retornos."""
        prices = pd.Series([100, 105, 110, 108, 115])
        returns = self.processor.calculate_returns(prices, method='simple')
        
        self.assertIsInstance(returns, pd.Series)
        self.assertEqual(len(returns), len(prices))
    
    def test_missing_data_detection(self):
        """Testa detecção de dados ausentes."""
        missing_info = self.processor.detect_missing_data(self.sample_data)
        
        self.assertIsInstance(missing_info, dict)
        self.assertIn('total_missing', missing_info)
        self.assertIn('columns_with_missing', missing_info)


class TestIntegration(unittest.TestCase):
    """Testes de integração entre módulos."""
    
    def test_end_to_end_analysis(self):
        """Testa análise completa end-to-end."""
        # Dados de exemplo
        data = pd.DataFrame({
            'receita_liquida': [1000000, 1100000, 1200000],
            'lucro_liquido': [100000, 120000, 140000],
            'ativo_total': [2000000, 2200000, 2400000],
            'patrimonio_liquido': [800000, 900000, 1000000],
            'ativo_circulante': [500000, 550000, 600000],
            'passivo_circulante': [200000, 220000, 240000],
            'ebitda': [200000, 220000, 240000],
            'fluxo_caixa_operacional': [150000, 160000, 170000],
            'fluxo_caixa_investimento': [-50000, -60000, -70000],
            'fluxo_caixa_financiamento': [-30000, -20000, -10000]
        })
        
        # Testar indicadores
        indicators = FinancialIndicators(data)
        roe = indicators.calculate_roe()
        self.assertGreater(roe, 0)
        
        # Testar valuation
        valuation = ValuationModels(data)
        dcf_result = valuation.dcf_analysis(taxa_desconto=0.10)
        self.assertGreater(dcf_result['valor_empresa'], 0)
        
        # Testar fluxo de caixa
        cash_flow = CashFlowAnalysis(data)
        fcf_result = cash_flow.calculate_free_cash_flow()
        self.assertIsInstance(fcf_result['valor_atual'], (int, float))


def run_tests():
    """Executa todos os testes."""
    # Criar suite de testes
    test_suite = unittest.TestSuite()
    
    # Adicionar testes
    test_classes = [
        TestFinancialIndicators,
        TestValuationModels,
        TestCashFlowAnalysis,
        TestMathematicalModels,
        TestFinancialUtils,
        TestDataCollector,
        TestDataProcessor,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Executar testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    if success:
        print("\n✅ Todos os testes passaram!")
    else:
        print("\n❌ Alguns testes falharam!")
        sys.exit(1)
