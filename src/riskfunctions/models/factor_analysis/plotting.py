"""Plotting helpers for the correlation heatmap, PCA scree graph, and loadings heatmaps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.decomposition as skd


def plot_correlation_heatmap(returns: pd.DataFrame) -> None:
    corr_matrix = returns.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()


def plot_pca_scree(pca: skd.PCA) -> None:
    x = range(1, len(pca.explained_variance_) + 1)
    plt.plot(x, pca.singular_values_, ls="dashed", lw=1)
    plt.scatter(x, pca.singular_values_, marker="o")
    plt.axhline(1, c="red", ls="dashed", lw=1)
    plt.ylim([0, 2.5])
    plt.title("Scree graph")
    plt.ylabel("Eigenvalue")
    plt.xlabel("Ordered eigenvalues")
    plt.show()


def plot_loadings_heatmap(loadings: pd.DataFrame, title: str) -> None:
    plt.figure(figsize=(25, 5))
    plt.title(title)
    ax = sns.heatmap(loadings.T, vmin=-1, vmax=1, center=0,
                      cmap=sns.diverging_palette(220, 20, n=200),
                      square=True, annot=True, fmt=".2f")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.show()
