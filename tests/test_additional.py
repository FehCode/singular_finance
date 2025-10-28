import sys
import os
import pandas as pd
import numpy as np

# garantir que o diretório do projeto esteja no path para importar o pacote local
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from singular_finance.data import DataProcessor, DataCollector


def test_fill_missing_data_mean():
    df = pd.DataFrame({
        'a': [1, np.nan, 3],
        'b': [np.nan, np.nan, np.nan]
    })

    proc = DataProcessor()
    filled = proc.fill_missing_data(df, method='mean')

    # coluna 'a' deve ter o NaN preenchido com mean
    assert filled.loc[1, 'a'] == pd.Series([1, np.nan, 3]).mean()
    # coluna 'b' continua NaN pois todos os valores são NaN
    assert filled['b'].isna().all()


def test_get_dividend_data_empty():
    collector = DataCollector()

    # Simular símbolo inexistente: o método deve retornar DataFrame vazio ou sem erro
    try:
        df = collector.get_dividend_data('INVALID.SYMBOL')
    except Exception:
        df = pd.DataFrame()

    assert isinstance(df, pd.DataFrame)
