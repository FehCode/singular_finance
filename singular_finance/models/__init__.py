"""
Módulo de Modelos Matemáticos Financeiros

Este módulo contém implementações de modelos matemáticos avançados
para análise financeira, incluindo modelos de precificação, otimização
e análise de risco.
"""

import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import differential_evolution, minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class MathematicalModels:
    """
    Classe principal para modelos matemáticos financeiros.

    Esta classe fornece implementações de diversos modelos matemáticos
    utilizados em análise financeira quantitativa.
    """

    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Inicializa a classe com dados financeiros.

        Args:
            data: DataFrame com dados financeiros (opcional)
        """
        self.data = data.copy() if data is not None else None
        self.models = {}  # Armazenar modelos treinados

    def black_scholes_option_pricing(
        self,
        spot_price: float,
        strike_price: float,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: float,
        option_type: str = "call",
    ) -> Dict[str, float]:
        """
        Calcula preço de opção usando modelo Black-Scholes.

        Args:
            spot_price: Preço atual do ativo subjacente
            strike_price: Preço de exercício
            time_to_maturity: Tempo até vencimento (em anos)
            risk_free_rate: Taxa livre de risco
            volatility: Volatilidade do ativo
            option_type: Tipo da opção ('call' ou 'put')

        Returns:
            Dicionário com preço da opção e gregas
        """
        # Calcular d1 e d2
        d1 = (
            np.log(spot_price / strike_price)
            + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity
        ) / (volatility * np.sqrt(time_to_maturity))

        d2 = d1 - volatility * np.sqrt(time_to_maturity)

        # Calcular preço da opção
        if option_type.lower() == "call":
            option_price = spot_price * stats.norm.cdf(d1) - strike_price * np.exp(
                -risk_free_rate * time_to_maturity
            ) * stats.norm.cdf(d2)
        else:  # put
            option_price = strike_price * np.exp(
                -risk_free_rate * time_to_maturity
            ) * stats.norm.cdf(-d2) - spot_price * stats.norm.cdf(-d1)

        # Calcular gregas
        delta = (
            stats.norm.cdf(d1)
            if option_type.lower() == "call"
            else stats.norm.cdf(d1) - 1
        )
        gamma = stats.norm.pdf(d1) / (
            spot_price * volatility * np.sqrt(time_to_maturity)
        )
        theta = (
            (
                -spot_price
                * stats.norm.pdf(d1)
                * volatility
                / (2 * np.sqrt(time_to_maturity))
                - risk_free_rate
                * strike_price
                * np.exp(-risk_free_rate * time_to_maturity)
                * stats.norm.cdf(d2)
            )
            if option_type.lower() == "call"
            else (
                -spot_price
                * stats.norm.pdf(d1)
                * volatility
                / (2 * np.sqrt(time_to_maturity))
                + risk_free_rate
                * strike_price
                * np.exp(-risk_free_rate * time_to_maturity)
                * stats.norm.cdf(-d2)
            )
        )
        vega = spot_price * stats.norm.pdf(d1) * np.sqrt(time_to_maturity)
        rho = (
            (
                strike_price
                * time_to_maturity
                * np.exp(-risk_free_rate * time_to_maturity)
                * stats.norm.cdf(d2)
            )
            if option_type.lower() == "call"
            else (
                -strike_price
                * time_to_maturity
                * np.exp(-risk_free_rate * time_to_maturity)
                * stats.norm.cdf(-d2)
            )
        )

        return {
            "option_price": option_price,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho,
            "d1": d1,
            "d2": d2,
        }

    def monte_carlo_simulation(
        self,
        initial_price: float,
        drift: float,
        volatility: float,
        time_horizon: float,
        num_simulations: int = 10000,
        num_steps: int = 252,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Simulação de Monte Carlo para preços de ativos.

        Args:
            initial_price: Preço inicial do ativo
            drift: Taxa de crescimento esperada
            volatility: Volatilidade anual
            time_horizon: Horizonte de tempo (em anos)
            num_simulations: Número de simulações
            num_steps: Número de passos de tempo

        Returns:
            Dicionário com resultados da simulação
        """
        dt = time_horizon / num_steps

        # Gerar números aleatórios
        random_shocks = np.random.normal(0, 1, (num_simulations, num_steps))

        # Calcular retornos
        returns = (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(
            dt
        ) * random_shocks

        # Calcular preços
        price_paths = np.zeros((num_simulations, num_steps + 1))
        price_paths[:, 0] = initial_price

        for t in range(num_steps):
            price_paths[:, t + 1] = price_paths[:, t] * np.exp(returns[:, t])

        # Calcular estatísticas
        final_prices = price_paths[:, -1]

        return {
            "price_paths": price_paths,
            "final_prices": final_prices,
            "mean_final_price": np.mean(final_prices),
            "std_final_price": np.std(final_prices),
            "percentile_5": np.percentile(final_prices, 5),
            "percentile_95": np.percentile(final_prices, 95),
            "var_95": initial_price - np.percentile(final_prices, 5),
            "expected_return": np.mean(final_prices) / initial_price - 1,
        }

    def portfolio_optimization(
        self,
        returns_data: pd.DataFrame,
        risk_free_rate: float = 0.02,
        target_return: Optional[float] = None,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Otimização de portfólio usando Markowitz.

        Args:
            returns_data: DataFrame com retornos históricos dos ativos
            risk_free_rate: Taxa livre de risco
            target_return: Retorno alvo (opcional)

        Returns:
            Dicionário com pesos otimizados e métricas
        """
        # Calcular matriz de covariância e retornos esperados
        cov_matrix = returns_data.cov().values
        expected_returns = returns_data.mean().values
        n_assets = len(expected_returns)

        # Função objetivo (minimizar variância)
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        # Restrições
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        ]  # Soma dos pesos = 1

        # Bounds (pesos entre 0 e 1)
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Otimização
        if target_return is not None:
            # Adicionar restrição de retorno alvo
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda x: np.dot(x, expected_returns) - target_return,
                }
            )

        result = minimize(
            objective,
            x0=np.ones(n_assets) / n_assets,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        optimal_weights = result.x

        # Calcular métricas do portfólio otimizado
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_variance = np.dot(
            optimal_weights.T, np.dot(cov_matrix, optimal_weights)
        )
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        return {
            "weights": optimal_weights,
            "expected_return": portfolio_return,
            "volatility": portfolio_volatility,
            "sharpe_ratio": sharpe_ratio,
            "variance": portfolio_variance,
        }

    def var_calculation(
        self,
        returns: pd.Series,
        confidence_level: float = 0.05,
        method: str = "historical",
    ) -> Dict[str, float]:
        """
        Calcula Value at Risk (VaR) usando diferentes métodos.

        Args:
            returns: Série com retornos históricos
            confidence_level: Nível de confiança (ex: 0.05 para 95%)
            method: Método de cálculo ('historical', 'parametric', 'monte_carlo')

        Returns:
            Dicionário com VaR e métricas relacionadas
        """
        returns_clean = returns.dropna()

        if method == "historical":
            # VaR histórico
            var = np.percentile(returns_clean, confidence_level * 100)

        elif method == "parametric":
            # VaR paramétrico (assumindo distribuição normal)
            mean_return = returns_clean.mean()
            std_return = returns_clean.std()
            var = mean_return + stats.norm.ppf(confidence_level) * std_return

        elif method == "monte_carlo":
            # VaR por simulação de Monte Carlo
            mean_return = returns_clean.mean()
            std_return = returns_clean.std()
            simulated_returns = np.random.normal(mean_return, std_return, 10000)
            var = np.percentile(simulated_returns, confidence_level * 100)

        else:
            raise ValueError(
                "Método deve ser 'historical', 'parametric' ou 'monte_carlo'"
            )

        # Calcular Expected Shortfall (CVaR)
        cvar = returns_clean[returns_clean <= var].mean()

        # Calcular métricas adicionais
        max_drawdown = self._calculate_max_drawdown(returns_clean)

        return {
            "var": var,
            "cvar": cvar,
            "confidence_level": confidence_level,
            "method": method,
            "max_drawdown": max_drawdown,
            "volatility": returns_clean.std(),
            "skewness": stats.skew(returns_clean),
            "kurtosis": stats.kurtosis(returns_clean),
        }

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calcula o máximo drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def cointegration_analysis(
        self, price_series: List[pd.Series], significance_level: float = 0.05
    ) -> Dict[str, Union[bool, float, np.ndarray]]:
        """
        Análise de cointegração entre séries de preços.

        Args:
            price_series: Lista de séries de preços
            significance_level: Nível de significância

        Returns:
            Dicionário com resultados da análise de cointegração
        """
        from statsmodels.tsa.stattools import coint

        if len(price_series) != 2:
            raise ValueError("Análise de cointegração requer exatamente 2 séries")

        series1, series2 = price_series

        # Teste de cointegração
        score, p_value, critical_values = coint(series1, series2)

        # Determinar se há cointegração
        is_cointegrated = p_value < significance_level

        # Calcular relação de cointegração
        if is_cointegrated:
            # Regressão para encontrar coeficiente de cointegração
            model = LinearRegression()
            model.fit(series1.values.reshape(-1, 1), series2.values)
            cointegration_coefficient = model.coef_[0]
            intercept = model.intercept_

            # Calcular spread
            spread = series2 - (cointegration_coefficient * series1 + intercept)

            # Teste de estacionariedade do spread
            from statsmodels.tsa.stattools import adfuller

            adf_stat, adf_pvalue, _, _, adf_critical, _ = adfuller(spread)

        else:
            cointegration_coefficient = None
            intercept = None
            spread = None
            adf_stat = None
            adf_pvalue = None

        return {
            "is_cointegrated": is_cointegrated,
            "p_value": p_value,
            "test_statistic": score,
            "critical_values": critical_values,
            "cointegration_coefficient": cointegration_coefficient,
            "intercept": intercept,
            "spread": spread,
            "spread_adf_stat": adf_stat,
            "spread_adf_pvalue": adf_pvalue,
        }

    def garch_model(
        self, returns: pd.Series, p: int = 1, q: int = 1
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Modelo GARCH para volatilidade.

        Args:
            returns: Série com retornos
            p: Número de termos ARCH
            q: Número de termos GARCH

        Returns:
            Dicionário com resultados do modelo GARCH
        """
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError("Biblioteca 'arch' é necessária para modelos GARCH")

        # Preparar dados
        returns_clean = returns.dropna() * 100  # Converter para percentual

        # Criar modelo GARCH
        model = arch_model(returns_clean, vol="Garch", p=p, q=q)

        # Estimar modelo
        fitted_model = model.fit(disp="off")

        # Previsões de volatilidade
        forecasts = fitted_model.forecast(horizon=1)
        forecasted_volatility = np.sqrt(forecasts.variance.iloc[-1, 0])

        return {
            "fitted_model": fitted_model,
            "forecasted_volatility": forecasted_volatility,
            "aic": fitted_model.aic,
            "bic": fitted_model.bic,
            "log_likelihood": fitted_model.loglikelihood,
            "parameters": fitted_model.params,
        }

    def machine_learning_forecast(
        self,
        data: pd.DataFrame,
        target_column: str,
        forecast_horizon: int = 1,
        model_type: str = "random_forest",
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Previsão usando machine learning.

        Args:
            data: DataFrame com dados históricos
            target_column: Nome da coluna alvo
            forecast_horizon: Horizonte de previsão
            model_type: Tipo do modelo ('random_forest', 'linear')

        Returns:
            Dicionário com previsões e métricas
        """
        # Preparar dados
        feature_columns = [col for col in data.columns if col != target_column]
        X = data[feature_columns].values
        y = data[target_column].values

        # Normalizar features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Dividir dados
        split_point = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_point], X_scaled[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]

        # Treinar modelo
        if model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "linear":
            model = LinearRegression()
        else:
            raise ValueError("Modelo deve ser 'random_forest' ou 'linear'")

        model.fit(X_train, y_train)

        # Fazer previsões
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        # Calcular métricas
        train_rmse = np.sqrt(np.mean((y_train - train_predictions) ** 2))
        test_rmse = np.sqrt(np.mean((y_test - test_predictions) ** 2))

        # Previsão futura (usando último ponto)
        last_features = X_scaled[-1:].reshape(1, -1)
        future_prediction = model.predict(last_features)[0]

        # Armazenar modelo
        self.models[f"{model_type}_{target_column}"] = {
            "model": model,
            "scaler": scaler,
            "feature_columns": feature_columns,
        }

        return {
            "train_predictions": train_predictions,
            "test_predictions": test_predictions,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "future_prediction": future_prediction,
            "model_type": model_type,
            "feature_importance": (
                model.feature_importances_
                if hasattr(model, "feature_importances_")
                else None
            ),
        }


def black_scholes_pricing(
    spot_price: float,
    strike_price: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call",
) -> Dict[str, float]:
    """
    Função utilitária para precificação Black-Scholes.

    Args:
        spot_price: Preço atual do ativo
        strike_price: Preço de exercício
        time_to_maturity: Tempo até vencimento
        risk_free_rate: Taxa livre de risco
        volatility: Volatilidade
        option_type: Tipo da opção

    Returns:
        Dicionário com preço e gregas
    """
    calculator = MathematicalModels()
    return calculator.black_scholes_option_pricing(
        spot_price,
        strike_price,
        time_to_maturity,
        risk_free_rate,
        volatility,
        option_type,
    )


def monte_carlo_simulation(
    initial_price: float,
    drift: float,
    volatility: float,
    time_horizon: float,
    num_simulations: int = 10000,
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Função utilitária para simulação Monte Carlo.

    Args:
        initial_price: Preço inicial
        drift: Taxa de crescimento
        volatility: Volatilidade
        time_horizon: Horizonte de tempo
        num_simulations: Número de simulações

    Returns:
        Dicionário com resultados da simulação
    """
    calculator = MathematicalModels()
    return calculator.monte_carlo_simulation(
        initial_price, drift, volatility, time_horizon, num_simulations
    )


def calculate_var(
    returns: pd.Series, confidence_level: float = 0.05, method: str = "historical"
) -> Dict[str, float]:
    """
    Função utilitária para cálculo de VaR.

    Args:
        returns: Série com retornos
        confidence_level: Nível de confiança
        method: Método de cálculo

    Returns:
        Dicionário com VaR e métricas
    """
    calculator = MathematicalModels()
    return calculator.var_calculation(returns, confidence_level, method)


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calcula o Índice de Força Relativa (RSI).
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """
    Calcula o MACD e a linha de sinal.
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return pd.DataFrame({"MACD": macd, "Signal": signal_line, "Hist": hist})


def calculate_sma(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Calcula Média Móvel Simples (SMA).
    """
    return prices.rolling(window=window).mean()


def calculate_ema(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Calcula Média Móvel Exponencial (EMA).
    """
    return prices.ewm(span=window, adjust=False).mean()


# Exemplos de padrões de candles simplificados (martelo e estrela cadente)
def is_hammer(
    open_prices: pd.Series,
    high_prices: pd.Series,
    low_prices: pd.Series,
    close_prices: pd.Series,
) -> pd.Series:
    """
    Detecta padrão de candle martelo.
    """
    body = abs(close_prices - open_prices)
    candle_range = high_prices - low_prices
    lower_shadow = open_prices - low_prices
    return (body < candle_range * 0.3) & (lower_shadow > body * 2)


def is_shooting_star(
    open_prices: pd.Series,
    high_prices: pd.Series,
    low_prices: pd.Series,
    close_prices: pd.Series,
) -> pd.Series:
    """
    Detecta padrão de estrela cadente.
    """
    body = abs(close_prices - open_prices)
    candle_range = high_prices - low_prices
    upper_shadow = high_prices - close_prices
    return (body < candle_range * 0.3) & (upper_shadow > body * 2)
