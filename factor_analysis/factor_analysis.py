"""
Factor Analysis: PCA (scree plot + loadings) and Factor Analysis with no
rotation, orthogonal (varimax) rotation, and oblique (oblimin) rotation.

Originally from Assignment_3_clean.ipynb, using an extended 10-asset
portfolio (equities across regions/sectors + oil + gold) intended to show
clear factor structure. Requires network access (yfinance),
scikit-learn, and `factor_analyzer` (pip install factor_analyzer).
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
import sklearn.decomposition as skd
from factor_analyzer import FactorAnalyzer


def load_extended_returns(start_date="2010-01-01", end_date="2024-01-01") -> pd.DataFrame:
    """
    Download an extended 10-asset portfolio designed to exhibit clear
    factor structure across region (EU/US), industry (aerospace/tech),
    and idiosyncratic (gold uncorrelated, oil broadly correlated) factors:
    EURO50, ASML, AIRBUS, S&P 500, AAPL, GM, Delta, Tencent, Oil, Gold.
    """
    stocks = ["SC0D.DE", "ASML.AS", "AIR.PA", "SPY", "AAPL", "GM", "DAL", "0700.HK", "CL=F", "GC=F"]
    stock_data = yf.download(" ".join(stocks), start=start_date, end=end_date)
    data = pd.DataFrame(
        stock_data["Adj Close"].values,
        columns=stock_data["Adj Close"].columns,
        index=stock_data.index,
    )
    data = data.rename(columns={
        "ASML.AS": "ASML", "AIR.PA": "AIRBUS", "SPY": "SNP500", "SC0D.DE": "EURO50",
        "DAL": "DELTA", "0700.HK": "TENCENT", "CL=F": "OIL", "GC=F": "GOLD",
    })
    returns = data.apply(np.log).diff(1).fillna(0)
    return returns


def plot_correlation_heatmap(returns: pd.DataFrame):
    corr_matrix = returns.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()


def run_pca_scree(returns: pd.DataFrame, n_components: int = 10):
    """Fit PCA with `n_components` and plot the scree graph of singular values."""
    pca = skd.PCA(n_components=n_components)
    pca.fit_transform(returns)

    plt.plot(range(1, len(pca.explained_variance_) + 1), pca.singular_values_, ls="dashed", lw=1)
    plt.scatter(range(1, len(pca.explained_variance_) + 1), pca.singular_values_, marker="o")
    plt.axhline(1, c="red", ls="dashed", lw=1)
    plt.ylim([0, 2.5])
    plt.title("Scree graph")
    plt.ylabel("Eigenvalue")
    plt.xlabel("Ordered eigenvalues")
    plt.show()

    return pca


def run_pca(returns: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
    """
    Fit PCA with the reduced number of components (chosen from the scree
    graph), and plot the loadings heatmap. Returns the loadings DataFrame.
    """
    pca = skd.PCA(n_components=n_components)
    pca.fit_transform(returns)

    loadings = pca.components_.T
    loadings_columns = [f"PC{i}" for i in range(n_components)]
    df_loadings = pd.DataFrame(loadings, columns=loadings_columns, index=returns.columns)

    plt.figure(figsize=(25, 5))
    plt.title("Correlation of each Principle Component with each Asset")
    ax = sns.heatmap(df_loadings.T, vmin=-1, vmax=1, center=0,
                      cmap=sns.diverging_palette(220, 20, n=200),
                      square=True, annot=True, fmt=".2f")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.show()

    return df_loadings


def run_factor_analysis(returns: pd.DataFrame, n_factors: int = 10, rotation: str | None = None, title: str = None) -> pd.DataFrame:
    """
    Fit a FactorAnalyzer model with the given rotation (None, 'varimax'
    for orthogonal, 'oblimin' for oblique), plot the loadings heatmap, and
    return the loadings DataFrame.
    """
    fa = FactorAnalyzer(n_factors, rotation=rotation)
    fa.fit(returns)

    fa_loadings = fa.loadings_
    fa_loadings_columns = [f"FA{i}" for i in range(1, n_factors + 1)]
    df_loadings_fa = pd.DataFrame(fa_loadings, columns=fa_loadings_columns, index=returns.columns)

    plt.figure(figsize=(25, 5))
    plt.title(title or f"Correlation of each Factor (rotation={rotation}) with each Asset")
    ax = sns.heatmap(df_loadings_fa.T, vmin=-1, vmax=1, center=0,
                      cmap=sns.diverging_palette(220, 20, n=200),
                      square=True, annot=True, fmt=".2f")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.show()

    return df_loadings_fa


def main():
    returns = load_extended_returns()
    plot_correlation_heatmap(returns)

    run_pca_scree(returns, n_components=10)
    run_pca(returns, n_components=5)

    run_factor_analysis(returns, n_factors=10, rotation=None, title="Correlation of each Factor with each Asset")
    run_factor_analysis(returns, n_factors=10, rotation="varimax", title="Correlation of each Factor with Orthogonal Rotation with each Asset")
    run_factor_analysis(returns, n_factors=10, rotation="oblimin", title="Correlation of each Factor with Oblique Rotation with each Asset")


if __name__ == "__main__":
    main()
